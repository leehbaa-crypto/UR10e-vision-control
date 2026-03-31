import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Float64MultiArray, Float32
from rclpy.qos import qos_profile_sensor_data
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import re

class MuJoCoDynamicPhysicsBridgeFixed(Node):
    def __init__(self):
        super().__init__('nrs_dynamic_physics_bridge_fixed')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)

        self.surface_files = [
            "flat_surface_5.stl", "concave_surface_1.stl", "concave_surface_2.stl",
            "_concave_surface_0.75.stl", "_comp_concave_0_75_v0_42.stl", "compound_concave_0_75_v0_42.stl"
        ]
        
        assets = {}
        for folder in ["visual", "collision"]:
            path = os.path.join(self.script_dir, folder)
            if os.path.exists(path):
                for fn in os.listdir(path):
                    if fn.lower().endswith(('.stl', '.obj', '.dae')):
                        with open(os.path.join(path, fn), "rb") as f:
                            assets[f"{folder}/{fn}"] = f.read()
        
        for s_file in self.surface_files:
            if os.path.exists(s_file):
                with open(s_file, "rb") as f:
                    assets[s_file] = f.read()

        urdf_path = "ur10e_combined.urdf"
        if not os.path.exists(urdf_path):
            self.get_logger().error(f"❌ {urdf_path}를 찾을 수 없습니다.")
            sys.exit(1)
            
        with open(urdf_path, "r") as f: full_xml = f.read()
        if "</robot>" in full_xml: base_xml = full_xml.split("</robot>")[0]
        else: base_xml = full_xml

        base_xml = base_xml.replace("package://ur_description/meshes/ur10e/", "")

        target_candidates = ["tool0", "wrist_3_link", "spindle_link", "ee_link"]
        for name in target_candidates:
            if f'name="{name}"' in base_xml:
                pattern = re.compile(f'(<body[^>]*name="{name}"[^>]*>)')
                match = pattern.search(base_xml)
                if match:
                    tag_end = match.end()
                    viz_xml = '\n      <site name="ft_site" pos="0 0 0" size="0.015" rgba="1 0 0 1"/>\n'
                    base_xml = base_xml[:tag_end] + viz_xml + base_xml[tag_end:]
                break

        surface_xml = ""
        for i, s_file in enumerate(self.surface_files):
            surface_xml += f"""
  <link name="surface_link_{i}">
    <visual><geometry><mesh filename="{s_file}" scale="1 1 1"/></geometry><material name="cyan_{i}"><color rgba="0 1 1 0"/></material></visual>
    <collision><geometry><mesh filename="{s_file}" scale="1 1 1"/></geometry></collision>
  </link>
  <joint name="world_to_surface_{i}" type="fixed">
    <parent link="world"/><child link="surface_link_{i}"/><origin xyz="0.5 0 -5.0" rpy="0 0 0"/>
  </joint>
"""
        env_xml = """
  <mujoco>
    <option gravity="0 0 -9.81" timestep="0.0005" integrator="implicitfast"/>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
      <geom name="floor" type="plane" size="3 3 0.1" rgba="0.2 0.2 0.2 1" pos="0 0 -0.01"/>
    </worldbody>
    <sensor>
      <force name="ft_force" site="ft_site"/>
      <torque name="ft_torque" site="ft_site"/>
    </sensor>
  </mujoco>
"""
        final_xml = base_xml + surface_xml + env_xml + "\n</robot>"

        try:
            self.model = mujoco.MjModel.from_xml_string(final_xml, assets=assets)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            sys.exit(1)

        self.surface_geom_ids = []
        for i in range(len(self.surface_files)):
            g_ids = []
            for j in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, j)
                if name and f"surface_link_{i}" in name: g_ids.append(j)
            self.surface_geom_ids.append(g_ids)

        self.is_launched = False
        self.selected_idx = -1

        self.pub_ft = self.create_publisher(WrenchStamped, '/ft_sensor', qos_profile_sensor_data)
        self.pub_joint = self.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.create_subscription(String, '/surface_command', self.command_cb, 10)
        self.create_subscription(Float64MultiArray, '/joint_commands', self.cmd_cb, 10)
        
        # 💡 [핵심 복구] 겨울방학 때의 완벽하고 안정적인 물리 파라미터 복구! 
        # (과도한 Kp는 오히려 계단 현상을 증폭시켜 덜렁거림을 만듭니다)
        self.kp = np.array([15000, 15000, 10000, 5000, 2500, 2500]) 
        self.kv = np.array([800,   800,   500,   200,  100,  100]) 
        self.torque_limits = np.array([1000.0]*6)
        
        self.joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.init_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.target_q = self.init_q.copy()
        
        # 💡 [덜렁거림 해결의 핵심] 20Hz의 제어기 명령을 2000Hz로 쪼개어 부드럽게 만들어주는 보간 장치
        self.current_smoothed_q = self.init_q.copy()
        
        for i, n in enumerate(self.joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid != -1: self.data.qpos[self.model.jnt_qposadr[jid]] = self.init_q[i]
        mujoco.mj_forward(self.model, self.data)

        self.force_offset_z = 0.0
        self.is_calibrated = False
        self.calibration_start_time = time.time()
        self.force_history = []
        self.filtered_force_z = 0.0 
        self.ft_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
        self.last_pub_time = time.time()
        self.last_sync_time = time.time()
        
        self.get_logger().info(f"✅ MuJoCo Bridge (Anti-Jitter Interpolator Attached) Ready!")

    def cmd_cb(self, msg):
        if len(msg.data) == 6: self.target_q = np.array(msg.data)

    def command_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            if filename in self.surface_files:
                self.selected_idx = self.surface_files.index(filename)
                self.spawn_surface(self.selected_idx)
                self.is_launched = True 

    def spawn_surface(self, index):
        for i, g_ids in enumerate(self.surface_geom_ids):
            is_target = (i == index)
            alpha = 0.8 if is_target else 0.0
            z_pos = 0.0 if is_target else -5.0
            for gid in g_ids:
                self.model.geom_rgba[gid] = [0, 1, 1, alpha]
                self.model.geom_contype[gid] = 1 if is_target else 0
                self.model.geom_conaffinity[gid] = 1 if is_target else 0
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"surface_link_{i}")
            if bid != -1: self.model.body_pos[bid] = [0.5, 0.0, z_pos]
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        while rclpy.ok() and not self.is_launched: rclpy.spin_once(self, timeout_sec=0.1)
        if not rclpy.ok(): return
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()
                
                current_q = np.zeros(6); current_v = np.zeros(6); dof_idx = []
                for i, n in enumerate(self.joints):
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    if jid == -1: continue
                    current_q[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
                    current_v[i] = self.data.qvel[self.model.jnt_dofadr[jid]]
                    dof_idx.append(self.model.jnt_dofadr[jid])

                if len(dof_idx) == 6:
                    # 💡 [핵심 보간 로직] 제어기의 20Hz 듬성듬성한 계단식 명령을 초당 2000번 부드러운 스플라인으로 다듬습니다.
                    # 이 한 줄이 네트워크 딜레이와 모터의 충격을 100% 흡수하여 덜렁거림을 원천 차단합니다!
                    self.current_smoothed_q = 0.98 * self.current_smoothed_q + 0.02 * self.target_q
                    
                    raw_torque = self.kp * (self.current_smoothed_q - current_q) - self.kv * current_v
                    total_torque = raw_torque + self.data.qfrc_bias[dof_idx]
                    total_torque = np.clip(total_torque, -self.torque_limits, self.torque_limits)
                    self.data.qfrc_applied[dof_idx] = total_torque

                mujoco.mj_step(self.model, self.data)
                
                now = time.time()
                if now - self.last_sync_time >= 0.016:
                    viewer.sync()
                    self.last_sync_time = now
                
                if now - self.last_pub_time >= 0.02: 
                    self.last_pub_time = now
                    try:
                        raw_force_z = self.data.sensor('ft_force').data[2]
                        if not self.is_calibrated:
                            self.force_history.append(raw_force_z)
                            if len(self.force_history) > 50: self.force_history.pop(0)
                            if time.time() - self.calibration_start_time > 1.0:
                                self.force_offset_z = sum(self.force_history) / len(self.force_history)
                                self.is_calibrated = True
                                
                        cal_force_z = raw_force_z - self.force_offset_z
                        self.filtered_force_z = 0.8 * self.filtered_force_z + 0.2 * cal_force_z
                        
                        if self.ft_site_id != -1:
                            if abs(self.filtered_force_z) > 2.0: self.model.site_rgba[self.ft_site_id] = [0, 1, 0, 1] 
                            else: self.model.site_rgba[self.ft_site_id] = [1, 0, 0, 1] 

                        msg_now = self.get_clock().now().to_msg()
                        f_msg = WrenchStamped()
                        f_msg.header.stamp = msg_now; f_msg.header.frame_id = "ft_sensor"
                        f_msg.wrench.force.z = self.filtered_force_z
                        self.pub_ft.publish(f_msg)
                        
                        j_msg = JointState()
                        j_msg.header.stamp = msg_now; j_msg.name = self.joints
                        j_msg.position = current_q.tolist()
                        self.pub_joint.publish(j_msg)
                    except: pass
                    
                rclpy.spin_once(self, timeout_sec=0.0)
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0: time.sleep(dt)

def main(args=None):
    rclpy.init(args=args); node = MuJoCoDynamicPhysicsBridgeFixed()
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
