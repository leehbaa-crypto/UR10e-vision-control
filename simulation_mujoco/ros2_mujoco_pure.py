import rclpy
from rclpy.node import Node
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import re

# MuJoCo UI 배율 설정 (150%)
os.environ["MUJOCO_UI_SCALE"] = "150"

class MinimalMuJoCoBridge(Node):
    def __init__(self, surface_file=None):
        super().__init__('minimal_mujoco_bridge')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(self.script_dir)
        
        # 1. 자산(Mesh) 로드
        assets = {}
        for folder in ["visual", "collision"]:
            path = os.path.join(self.script_dir, folder)
            if os.path.exists(path):
                for fn in os.listdir(path):
                    if fn.lower().endswith(('.stl', '.obj', '.dae')):
                        with open(os.path.join(path, fn), "rb") as f:
                            assets[f"{folder}/{fn}"] = f.read()

        # 2. 선택된 Surface가 있다면 자산에 추가
        if surface_file and os.path.exists(surface_file):
            with open(surface_file, "rb") as f:
                assets[surface_file] = f.read()

        # 3. URDF 읽기 및 수정
        urdf_files = [f for f in os.listdir('.') if f.endswith('.urdf')]
        if not urdf_files: sys.exit(1)
        
        with open(urdf_files[0], "r") as f: 
            urdf_content = f.read()
        
        urdf_content = urdf_content.replace("package://ur_description/meshes/ur10e/", "")
        
        # Surface 소환 로직 (기존 bridge 로직 이식)
        if surface_file:
            # workpiece_link가 있다면 제거하고 다시 생성
            if "<link name=\"workpiece_link\">" in urdf_content:
                urdf_content = urdf_content.split("<link name=\"workpiece_link\">")[0] + "</robot>"
            
            surface_xml = f"""
  <link name="workpiece_link">
    <visual><geometry><mesh filename="{surface_file}" scale="1 1 1"/></geometry><material name="blue"><color rgba="0 0 1 0.7"/></material></visual>
    <collision><geometry><mesh filename="{surface_file}" scale="1 1 1"/></geometry></collision>
  </link>
  <joint name="world_to_workpiece" type="fixed">
    <parent link="world"/><child link="workpiece_link"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
"""
            urdf_content = urdf_content.replace("</robot>", surface_xml + "</robot>")

        # 4. MuJoCo 모델 로드
        try:
            self.model = mujoco.MjModel.from_xml_string(urdf_content, assets=assets)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"❌ MuJoCo 모델 로드 실패: {e}")
            sys.exit(1)

        # 관절 초기화 및 중력 보상 설정
        self.init_q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        for i, joint_name in enumerate(self.joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid != -1: self.data.qpos[self.model.jnt_qposadr[jid]] = self.init_q[i]
        mujoco.mj_forward(self.model, self.data)
        
        self.get_logger().info(f"🤖 Simulation Started with Surface: {surface_file}")

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running() and rclpy.ok():
                step_start = time.time()
                self.data.qfrc_applied[:] = self.data.qfrc_bias[:]
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                rclpy.spin_once(self, timeout_sec=0.0)
                dt = self.model.opt.timestep - (time.time() - step_start)
                if dt > 0: time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    # 명령행 인자에서 파일명 받기
    surface_file = sys.argv[1] if len(sys.argv) > 1 else None
    bridge_node = MinimalMuJoCoBridge(surface_file)
    try:
        bridge_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        bridge_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
