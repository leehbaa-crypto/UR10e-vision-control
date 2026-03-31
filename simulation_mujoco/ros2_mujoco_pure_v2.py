import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import re
from std_msgs.msg import String, Float64MultiArray

class MuJoCoDynamicSurfaceBridge(Node):
    def __init__(self):
        super().__init__('nrs_dynamic_surface_bridge')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)

        # 1. Surface 파일 리스트
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

        # 3. URDF 로드 및 동적 조립
        urdf_path = "ur10e_combined.urdf"
        if not os.path.exists(urdf_path):
            self.get_logger().error(f"❌ {urdf_path}를 찾을 수 없습니다.")
            sys.exit(1)
            
        with open(urdf_path, "r") as f:
            full_xml = f.read()
        
        # [중요] 기존 workpiece_link와 joint가 포함된 부분을 완전히 무시하고
        # 로봇 정의 부분(world부터 wrist_3까지)만 추출하거나, 깔끔하게 잘라냅니다.
        # 가장 안전한 방법: </robot> 태그 앞까지만 쓰고 내가 직접 닫기
        if "</robot>" in full_xml:
            base_xml = full_xml.split("</robot>")[0]
        else:
            base_xml = full_xml

        # 경로 수정
        base_xml = base_xml.replace("package://ur_description/meshes/ur10e/", "")

        # 동적 표면 XML 생성
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
        # MuJoCo 전용 환경 설정
        env_xml = """
  <mujoco>
    <worldbody>
      <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
      <geom name="floor" type="plane" size="3 3 0.1" rgba="0.2 0.2 0.2 1" pos="0 0 -0.01"/>
    </worldbody>
  </mujoco>
"""
        # 최종 합치기
        final_xml = base_xml + surface_xml + env_xml + "\n</robot>"

        # 4. 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_string(final_xml, assets=assets)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"❌ 모델 생성 실패: {e}")
            # 에러 디버깅을 위해 결과물 저장
            with open("failed_model.xml", "w") as f: f.write(final_xml)
            sys.exit(1)

        # 5. 초기 포즈 설정 (ㄱ 자)
        self.init_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.target_q = self.init_q.copy()
        
        # 6. 표면 Geom ID 추출
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

        # 7. ROS 2 구독 설정
        self.create_subscription(String, '/surface_command', self.command_cb, 10)
        self.create_subscription(Float64MultiArray, '/joint_commands', self.joint_cb, 10)
        
        self.get_logger().info("✅ MuJoCo bridge initialized (Fixed XML structure).")

    def joint_cb(self, msg):
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
        self.get_logger().info(f"✨ Surface Active: {self.surface_files[index]}")

    def run(self):
        while rclpy.ok() and not self.is_launched:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if not rclpy.ok(): return
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()
                self.data.qpos[:6] = self.target_q
                self.data.qvel[:6] = 0.0
                self.data.qfrc_applied[:] = self.data.qfrc_bias[:]
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                rclpy.spin_once(self, timeout_sec=0.0)
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0: time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoDynamicSurfaceBridge()
    try: node.run()
    except KeyboardInterrupt: pass
    finally:
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
