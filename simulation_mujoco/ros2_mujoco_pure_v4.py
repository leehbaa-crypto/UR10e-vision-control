import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String, Float64MultiArray
from rclpy.qos import qos_profile_sensor_data
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import re

class MuJoCoDynamicPhysicsBridgeV5(Node):
    def __init__(self):
        super().__init__('nrs_dynamic_physics_bridge_v5')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)

        # 1. Surface 파일 리스트 (동적 소환용)
        self.surface_files = [
            "flat_surface_5.stl",
            "concave_surface_1.stl",
            "concave_surface_2.stl",
            "_concave_surface_0.75.stl",
            "_comp_concave_0_75_v0_42.stl",
            "compound_concave_0_75_v0_42.stl"
        ]
        
        # 2. 자산 로드
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

        # 3. URDF 로드 및 전처리
        urdf_path = "ur10e_combined.urdf"
        if not os.path.exists(urdf_path):
            self.get_logger().error(f"❌ {urdf_path}를 찾을 수 없습니다.")
            sys.exit(1)
            
        with open(urdf_path, "r") as f:
            full_xml = f.read()
        
        if "</robot>" in full_xml:
            base_xml = full_xml.split("</robot>")[0]
        else:
            base_xml = full_xml

        base_xml = base_xml.replace("package://ur_description/meshes/ur10e/", "")

        # 💡 [과거 브릿지 복구 1] F/T 센서 사이트를 로봇 끝단에 동적으로 삽입
        target_candidates = ["tool0", "wrist_3_link", "spindle_link", "ee_link"]
        found_body = None
        for name in target_candidates:
            if f'name="{name}"' in base_xml:
                found_body = name
                pattern = re.compile(f'(<body[^>]*name="{name}"[^>]*>)')
                match = pattern.search(base_xml)
                if match:
                    tag_end = match.end()
                    viz_xml = '\n      <site name="ft_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>\n'
                    base_xml = base_xml[:tag_end] + viz_xml + base_xml[tag_end:]
                break

        # 4. 동적 표면 XML 생성
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
        # 💡 [과거 브릿지 복구 2] 덜덜거림을 완벽히 잡는 timestep=0.0005 와 F/T 센서 블록 삽입!
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
            self.get_logger().error(f"❌ 모델 생성 실패: {e}")
            sys.exit(1)

        self.surface_geom_ids = []
        for i in range(len(self.surface_files)):
            g_ids = []
            for j in range(self.model.ngeom):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, j)
                if name and f"surface_link_{i}" in name:
                    g_ids.append(j)
            self.surface_geom_ids.append(g_ids)

        self.is_launched = False
        self.selected_idx = -1

        # 5. ROS 2 구독/발행 설정 (과거 센서 출력 완벽 지원)
        self.pub_ft = self.create_publisher(WrenchStamped, '/ft_sensor', qos_profile_sensor_data)
        self.pub_joint = self.create_publisher(JointState, '/joint_states', qos_profile_sensor_data)
        self.create_subscription(String, '/surface_command', self.command_cb, 10)
        self.create_subscription(Float64MultiArray, '/joint_commands', self.cmd_cb, 10)
        
        # 💡 [과거 브릿지 복구 3] 튜닝 깎아두신 PID 게인 완벽 복구
        self.kp = np.array([15000, 15000, 10000, 800, 800, 800]) 
        self.kv = np.array([800,   800,   500,   30,  30,  30]) 
        self.torque_limits = np.array([1000.0]*6)
        
        self.joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.init_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.target_q = self.init_q.copy()
        
        for i, n in enumerate(self.joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid != -1: self.data.qpos[self.model.jnt_qposadr[jid]] = self.init_q[i]
        mujoco.mj_forward(self.model, self.data)

        self.force_offset_z = 0.0
        self.is_calibrated = False
        self.calibration_start_time = time.time()
        self.force_history = []
        
        self.get_logger().info(f"✅ MuJoCo Bridge V5 (Pure + Dynamics) Ready! [FT_Site: {found_body}]")

    def cmd_cb(self, msg):
        if len(msg.data) == 6:
            self.target_q = np.array(msg.data)

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
        self.get_logger().info(f"✨ Surface Active & Collisions Enabled: {self.surface_files[index]}")

    def run(self):
        while rclpy.ok() and not self.is_launched:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not rclpy.ok(): return
        
        last_log = time.time()
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()
                
                # 💡 [핵심] 위치 순간이동(qpos=target)을 삭제하고 토크 구동으로 완벽 복구
                current_q = np.zeros(6); current_v = np.zeros(6); dof_idx = []
                for i, n in enumerate(self.joints):
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    if jid == -1: continue
                    current_q[i] = self.data.qpos[self.model.jnt_qposadr[jid]]
                    current_v[i] = self.data.qvel[self.model.jnt_dofadr[jid]]
                    dof_idx.append(self.model.jnt_dofadr[jid])

                if len(dof_idx) == 6:
                    raw_torque = self.kp * (self.target_q - current_q) - self.kv * current_v
                    total_torque = raw_torque + self.data.qfrc_bias[dof_idx]
                    total_torque = np.clip(total_torque, -self.torque_limits, self.torque_limits)
                    self.data.qfrc_applied[dof_idx] = total_torque

                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # 💡 [센서 출력] 과거 브릿지처럼 /ft_sensor와 /joint_states 를 50Hz로 뿜어냅니다.
                if int(time.time()*100) % 5 == 0:
                    try:
                        now = self.get_clock().now().to_msg()
                        
                        raw_force_z = self.data.sensor('ft_force').data[2]
                        self.force_history.append(raw_force_z)
                        if len(self.force_history) > 10: self.force_history.pop(0)
                        avg_force_z = sum(self.force_history) / len(self.force_history)
                        
                        if not self.is_calibrated:
                            if time.time() - self.calibration_start_time > 1.0:
                                self.force_offset_z = avg_force_z
                                self.is_calibrated = True
                        cal_force_z = avg_force_z - self.force_offset_z
                        
                        f_msg = WrenchStamped()
                        f_msg.header.stamp = now; f_msg.header.frame_id = "ft_sensor"
                        f_msg.wrench.force.z = cal_force_z
                        self.pub_ft.publish(f_msg)
                        
                        j_msg = JointState()
                        j_msg.header.stamp = now; j_msg.name = self.joints
                        j_msg.position = current_q.tolist()
                        self.pub_joint.publish(j_msg)
                        
                        if time.time() - last_log > 2.0:
                            print(f"📡 Force: {cal_force_z:.2f}N | Tracking")
                            last_log = time.time()
                    except: pass
                    
                rclpy.spin_once(self, timeout_sec=0.0)
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0: time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoDynamicPhysicsBridgeV5()
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
