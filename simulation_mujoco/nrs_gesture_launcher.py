import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import os
import sys

class GestureLauncher(Node):
    def __init__(self):
        super().__init__('nrs_gesture_launcher')
        
        # /surface_command 토픽 구독 (SPAWN:파일명 메시지 대기)
        self.create_subscription(String, '/surface_command', self.command_cb, 10)
        
        self.get_logger().info("="*50)
        self.get_logger().info(" 🖐️ Gesture-Based Surface Launcher Ready!")
        self.get_logger().info(" 1. Use camera to select a surface.")
        self.get_logger().info(" 2. Make a 'ROCK' gesture with RIGHT hand to launch MuJoCo.")
        self.get_logger().info("="*50)

    def command_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            self.get_logger().info(f"🚀 Final Confirmation Received! Launching MuJoCo with {filename}...")
            
            # 현재 스크립트 위치 확인
            script_path = os.path.join(os.path.expanduser("~"), "2026-1_urp/simulation_mujoco/ros2_mujoco_pure.py")
            
            # MuJoCo 브릿지 실행 (서브프로세스)
            try:
                # 여기서 실행하면 이 런처는 MuJoCo가 닫힐 때까지 대기하게 됩니다.
                subprocess.run(["python3", script_path, filename])
            except Exception as e:
                self.get_logger().error(f"Failed to launch MuJoCo: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GestureLauncher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
