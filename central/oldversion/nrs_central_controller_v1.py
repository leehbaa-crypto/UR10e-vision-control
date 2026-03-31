import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time

class CentralControllerV1(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v1')
        
        # 1. 상태 정의 (FSM)
        self.STATE_NOT_READY = "NOT READY"
        self.STATE_IDLE = "IDLE"
        self.STATE_FOLLOWING = "FOLLOWING"
        self.STATE_ESTOP = "ESTOP"
        self.STATE_RESETTING = "RESETTING"
        self.current_state = self.STATE_NOT_READY
        
        # 2. 제어 및 필터 변수
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.filtered_target = self.target_pose.copy()
        self.alpha = 0.6 # [개선] 반응 속도 대폭 향상 (0.3 -> 0.6)
        
        # 3. 안전 제약 조건 (Safety Constraints)
        self.q_min = np.array([-6.28, -3.14, -3.14, -6.28, -6.28, -6.28])
        self.q_max = np.array([ 6.28,  0.00,  3.14,  6.28,  6.28,  6.28])
        self.x_range = [0.3, 1.0]   
        self.y_range = [-0.6, 0.6]  
        self.z_min = 0.05           

        # 4. DH Parameters (UR10e)
        self.dh_params = [
            {'a': 0,        'd': 0.1807,  'alpha': 1.570796},
            {'a': -0.6127,  'd': 0,       'alpha': 0},
            {'a': -0.57155, 'd': 0,       'alpha': 0},
            {'a': 0,        'd': 0.17415, 'alpha': 1.570796},
            {'a': 0,        'd': 0.11985, 'alpha': -1.570796},
            {'a': 0,        'd': 0.11655, 'alpha': 0}
        ]
        
        # 5. 제스처 및 타이머 관리
        self.last_log_time = 0
        self.gesture_start_time = None
        self.required_duration = 2.0 
        self.last_valid_target_time = time.time()
        self.is_first_follow_step = True
        self.current_gestures = {"User_Left": "None", "User_Right": "None"}

        # 6. ROS 2 인터페이스
        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.timer = self.create_timer(0.05, self.control_loop) 
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.get_logger().info("🚀 Optimized Central Controller V1 Initialized.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            self.current_state = self.STATE_IDLE

    def target_cb(self, msg):
        raw_target = np.array([msg.z + 0.2, -msg.x, -msg.y + 0.8])
        raw_target[0] = np.clip(raw_target[0], self.x_range[0], self.x_range[1])
        raw_target[1] = np.clip(raw_target[1], self.y_range[0], self.y_range[1])
        raw_target[2] = np.max([raw_target[2], self.z_min])
        
        # [EMA Filter] 필터 계수 0.6으로 기민한 반응 확보
        self.filtered_target = self.alpha * raw_target + (1.0 - self.alpha) * self.filtered_target
        self.target_pose = self.filtered_target
        
        now = time.time()
        if now - self.last_log_time > 0.5:
            print(f"📍 [V1] Target: {self.target_pose[0]:.2f}, {self.target_pose[1]:.2f}, {self.target_pose[2]:.2f}")
            self.last_log_time = now
        self.last_valid_target_time = now

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        try:
            parts = msg.data.split(',')
            self.current_gestures["User_Right"] = parts[0].split(':')[1]
            self.current_gestures["User_Left"] = parts[1].split(':')[1]
            self.update_state_machine()
        except: pass

    def update_state_machine(self):
        l, r = self.current_gestures["User_Left"], self.current_gestures["User_Right"]
        now = time.time()

        if l == "rock" and r == "rock":
            if self.current_state not in [self.STATE_ESTOP, self.STATE_NOT_READY]:
                self.current_state = self.STATE_ESTOP
                self.get_logger().warn("🚨 EMERGENCY STOP")
            return

        if l == "paper" and r == "rock":
            if self.current_state in [self.STATE_IDLE, self.STATE_NOT_READY]:
                if self.gesture_start_time is None: self.gesture_start_time = now
                elif now - self.gesture_start_time >= self.required_duration:
                    self.current_state = self.STATE_FOLLOWING
                    self.get_logger().info("▶️ START FOLLOWING")
                    self.last_valid_target_time = now
                    self.is_first_follow_step = True
                    self.gesture_start_time = None
            else: self.gesture_start_time = None
        
        elif l == "paper" and r == "paper":
            if self.current_state in [self.STATE_IDLE, self.STATE_ESTOP, self.STATE_FOLLOWING]:
                if self.gesture_start_time is None: self.gesture_start_time = now
                elif now - self.gesture_start_time >= self.required_duration:
                    self.current_state = self.STATE_RESETTING
                    self.gesture_start_time = None
            else: self.gesture_start_time = None
        else:
            self.gesture_start_time = None

    def numerical_ik(self, target, current_q):
        """
        [핵심 최적화] 기하학적 자코비안 기반 고성능 IK.
        FK 호출을 획기적으로 줄여 CPU 부하를 제거하고 제어 루프 끊김을 해결함.
        """
        q = current_q.copy()
        for _ in range(10):
            # 1. Forward Kinematics (중간 변환 행렬 및 조인트 위치 저장)
            T = np.eye(4)
            z_axes = [np.array([0, 0, 1])] # Base Z축
            p_joints = [np.zeros(3)]      # Base 위치
            
            for i in range(6):
                a, d, alpha = self.dh_params[i]['a'], self.dh_params[i]['d'], self.dh_params[i]['alpha']
                ct, st = np.cos(q[i]), np.sin(q[i]); ca, sa = np.cos(alpha), np.sin(alpha)
                Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
                T = T @ Ti
                z_axes.append(T[:3, 2])
                p_joints.append(T[:3, 3])
            
            p_ee = T[:3, 3]
            err = target - p_ee
            if np.linalg.norm(err) < 1e-3: break
            
            # 2. 기하학적 자코비안 (Revolute joints) 직접 구성
            J = np.zeros((3, 6))
            for i in range(6):
                J[:, i] = np.cross(z_axes[i], (p_ee - p_joints[i]))
            
            # 3. Damped Least Squares (특이점 회피 및 안정적 수렴)
            lambda_sq = 0.001
            step = J.T @ np.linalg.inv(J @ J.T + lambda_sq * np.eye(3)) @ err
            
            q += np.clip(step, -0.2, 0.2) # 기민하게 움직이도록 상향
            q = np.clip(q, self.q_min, self.q_max) # 관절 보호
        return q

    def control_loop(self):
        if self.current_state == self.STATE_NOT_READY: return
        now = time.time()
        cmd = Float64MultiArray()
        
        if self.current_state == self.STATE_FOLLOWING:
            if self.is_first_follow_step:
                if now - self.last_valid_target_time < 0.1: self.is_first_follow_step = False
                else: self.target_q = self.current_joints
            
            if not self.is_first_follow_step and (now - self.last_valid_target_time > 2.0):
                self.current_state = self.STATE_ESTOP
                self.get_logger().error("⚠️ Tracking Lost!")
            else:
                self.target_q = self.numerical_ik(self.target_pose, self.current_joints)
                
        elif self.current_state == self.STATE_RESETTING:
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = self.STATE_IDLE
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralControllerV1()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
