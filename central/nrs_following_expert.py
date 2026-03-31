import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time

class FollowingExpert(Node):
    def __init__(self):
        super().__init__('nrs_following_expert')
        
        # 1. 상태 및 제어 변수 (선행연구 비법 반영)
        self.current_state = "NOT READY"
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        # 필터 및 목표 좌표
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.filtered_target = self.target_pose.copy()
        self.last_sent_target = self.target_pose.copy()
        
        # [비법 1] 가변 필터 계수 및 데드존
        self.alpha = 0.4        # 필터 강도
        self.deadzone = 0.005   # 5mm 미만 움직임 무시 (떨림 방지)
        self.z_offset = 0.03    # [선행연구] 표면 위 3cm 부유
        
        # [비법 2] 안전 영역 및 관절 한계
        self.q_min = np.array([-6.28, -3.14, -3.14, -6.28, -6.28, -6.28])
        self.q_max = np.array([ 6.28,  0.00,  3.14,  6.28,  6.28,  6.28])
        self.workspace = {'x': [0.35, 0.9], 'y': [-0.5, 0.5], 'z': [0.05, 0.6]}

        # DH Params (UR10e)
        self.dh = [
            {'a': 0,        'd': 0.1807,  'alpha': 1.570796},
            {'a': -0.6127,  'd': 0,       'alpha': 0},
            {'a': -0.57155, 'd': 0,       'alpha': 0},
            {'a': 0,        'd': 0.17415, 'alpha': 1.570796},
            {'a': 0,        'd': 0.11985, 'alpha': -1.570796},
            {'a': 0,        'd': 0.11655, 'alpha': 0}
        ]

        # 2. ROS 2 설정
        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # [비법 3] 30Hz 고정 주기로 고속 제어 (0.033s)
        self.timer = self.create_timer(0.033, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.last_valid_time = time.time()
        
        self.get_logger().info("🌟 Following Expert (Legacy Wisdom + ROS 2 Fast IK) Ready.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"): self.current_state = "IDLE"

    def target_cb(self, msg):
        # 좌표 변환 및 Z-Offset 적용
        raw_x = np.clip(msg.z + 0.2, self.workspace['x'][0], self.workspace['x'][1])
        raw_y = np.clip(-msg.x, self.workspace['y'][0], self.workspace['y'][1])
        raw_z = np.clip(-msg.y + 0.8 + self.z_offset, self.workspace['z'][0], self.workspace['z'][1])
        
        new_target = np.array([raw_x, raw_y, raw_z])
        
        # [데드존 로직] 미세한 움직임은 필터링에서 제외
        if np.linalg.norm(new_target - self.last_sent_target) > self.deadzone:
            self.filtered_target = self.alpha * new_target + (1.0 - self.alpha) * self.filtered_target
            self.target_pose = self.filtered_target
            self.last_sent_target = new_target
            self.last_valid_time = time.time()

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        parts = msg.data.split(',')
        r, l = parts[0].split(':')[1], parts[1].split(':')[1]
        if l == "rock" and r == "rock": self.current_state = "ESTOP"
        elif l == "paper" and r == "rock": self.current_state = "FOLLOWING"
        elif l == "paper" and r == "paper": self.current_state = "RESETTING"

    def fast_geometric_ik(self, target, current_q):
        """ [비법 4] 고속 기하학적 자코비안 엔진 """
        q = current_q.copy()
        for _ in range(8): # 8회 반복으로도 충분한 정밀도
            T = np.eye(4); z_axes = [np.array([0, 0, 1])]; p_joints = [np.zeros(3)]
            for i in range(6):
                a, d, alpha = self.dh[i]['a'], self.dh[i]['d'], self.dh[i]['alpha']
                ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(alpha), np.sin(alpha)
                Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
                T = T @ Ti
                z_axes.append(T[:3, 2]); p_joints.append(T[:3, 3])
            
            p_ee = T[:3, 3]; err = target - p_ee
            if np.linalg.norm(err) < 0.001: break
            
            J = np.zeros((3, 6))
            for i in range(6): J[:, i] = np.cross(z_axes[i], (p_ee - p_joints[i]))
            
            # Damped Pseudo-inverse
            step = J.T @ np.linalg.inv(J @ J.T + 0.001 * np.eye(3)) @ err
            q += np.clip(step, -0.2, 0.2)
            q = np.clip(q, self.q_min, self.q_max)
        return q

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()
        
        if self.current_state == "FOLLOWING":
            if time.time() - self.last_valid_time > 1.5:
                self.current_state = "ESTOP"
            else:
                self.target_q = self.fast_geometric_ik(self.target_pose, self.current_joints)
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = FollowingExpert()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
