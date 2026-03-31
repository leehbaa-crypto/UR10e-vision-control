import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time

class CentralController(Node):
    def __init__(self):
        super().__init__('nrs_central_controller')
        
        # 1. 상태 정의
        self.STATE_NOT_READY = "NOT READY"
        self.STATE_IDLE = "IDLE"
        self.STATE_FOLLOWING = "FOLLOWING"
        self.STATE_ESTOP = "ESTOP"
        self.STATE_RESETTING = "RESETTING"
        self.current_state = self.STATE_NOT_READY
        
        # 2. 제어 변수
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        # [안전] 관절 한계 (UR10e 규격 기준, 안전을 위해 조금 더 좁게 설정)
        self.q_min = np.array([-6.28, -3.14, -3.14, -6.28, -6.28, -6.28])
        self.q_max = np.array([ 6.28,  0.00,  3.14,  6.28,  6.28,  6.28])

        # [안전] 작업 영역 제한 (로봇 베이스 기준 미터 단위)
        self.x_range = [0.3, 1.0]   # 너무 가깝거나 멀지 않게
        self.y_range = [-0.6, 0.6]  # 좌우 범위
        self.z_min = 0.05           # 바닥 뚫기 방지 (5cm 여유)

        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.filtered_target = self.target_pose.copy()
        
        self.last_valid_target_time = time.time()
        self.is_first_follow_step = True 
        
        # DH Params (UR10e)
        self.dh_params = [
            {'a': 0,        'd': 0.1807,  'alpha': 1.570796},
            {'a': -0.6127,  'd': 0,       'alpha': 0},
            {'a': -0.57155, 'd': 0,       'alpha': 0},
            {'a': 0,        'd': 0.17415, 'alpha': 1.570796},
            {'a': 0,        'd': 0.11985, 'alpha': -1.570796},
            {'a': 0,        'd': 0.11655, 'alpha': 0}
        ]
        
        # 3. 제스처 관리
        self.last_log_time = 0
        self.gesture_start_time = None
        self.required_duration = 2.0 
        self.current_gestures = {"User_Left": "None", "User_Right": "None"}

        # 4. ROS 2 설정
        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 20Hz (0.05s) 주기 유지
        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.get_logger().info(f"🚀 Light-weight Controller Ready.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            self.current_state = self.STATE_IDLE
            self.get_logger().info("✅ Simulation IDLE")

    def target_cb(self, msg):
        # 1. 좌표 변환
        raw_target = np.array([msg.z + 0.2, -msg.x, -msg.y + 0.8])
        
        # 2. [안전] 작업 영역 클램핑 (로봇이 이상한 곳으로 튀는 것 방지)
        raw_target[0] = np.clip(raw_target[0], self.x_range[0], self.x_range[1])
        raw_target[1] = np.clip(raw_target[1], self.y_range[0], self.y_range[1])
        raw_target[2] = np.max([raw_target[2], self.z_min]) # 바닥 보호
        
        # 3. 필터링
        self.filtered_target = self.alpha * raw_target + (1.0 - self.alpha) * self.filtered_target
        self.target_pose = self.filtered_target
        
        # 4. 로그
        now = time.time()
        if now - self.last_log_time > 0.5:
            current_p = self.get_fk(self.current_joints)
            dist_err = np.linalg.norm(self.target_pose - current_p)
            print(f"📍 Target: {self.target_pose[0]:.2f}, {self.target_pose[1]:.2f}, {self.target_pose[2]:.2f} | Dist Err: {dist_err:.3f}m")
            self.last_log_time = now
            
        self.last_valid_target_time = now

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        try:
            parts = msg.data.split(',')
            self.current_gestures["User_Right"] = parts[0].split(':')[1]
            self.current_gestures["User_Left"] = parts[1].split(':')[1]
            
            # 상태 변경 시 로그 출력을 위해 update_state_machine 호출 전후 체크 가능
            self.update_state_machine()
        except: pass

    def update_state_machine(self):
        l, r = self.current_gestures["User_Left"], self.current_gestures["User_Right"]
        now = time.time()

        if l == "rock" and r == "rock":
            if self.current_state not in [self.STATE_ESTOP, self.STATE_NOT_READY]:
                self.current_state = self.STATE_ESTOP
                self.get_logger().warn("🚨 EMERGENCY STOP: 'Rock-Rock' Gesture Detected")
            return

        if l == "paper" and r == "rock":
            if self.current_state in [self.STATE_IDLE, self.STATE_NOT_READY]:
                if self.gesture_start_time is None: self.gesture_start_time = now
                elif now - self.gesture_start_time >= self.required_duration:
                    self.current_state = self.STATE_FOLLOWING
                    self.get_logger().info("▶️ START FOLLOWING (Target: Right Index Tip)")
                    self.last_valid_target_time = time.time()
                    self.is_first_follow_step = True 
                    self.gesture_start_time = None
            else: self.gesture_start_time = None
        # ... (중략)
        
        elif l == "paper" and r == "paper":
            if self.current_state in [self.STATE_IDLE, self.STATE_ESTOP, self.STATE_FOLLOWING]:
                if self.gesture_start_time is None: self.gesture_start_time = now
                elif now - self.gesture_start_time >= self.required_duration:
                    self.current_state = self.STATE_RESETTING
                    self.gesture_start_time = None
            else: self.gesture_start_time = None
        else:
            self.gesture_start_time = None

    def get_fk(self, q):
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = self.dh_params[i]['a'], self.dh_params[i]['d'], self.dh_params[i]['alpha']
            ct, st = np.cos(q[i]), np.sin(q[i]); ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
            T = T @ Ti
        return T[:3, 3]

    def numerical_ik(self, target, current_q):
        q = current_q.copy()
        # [복구] 반복 횟수 10회로 증가, 정밀도 확보
        for _ in range(10):
            p = self.get_fk(q)
            err = target - p
            if np.linalg.norm(err) < 1e-3: break
            
            J = np.zeros((3, 6))
            delta = 1e-5
            for i in range(6):
                qp = q.copy(); qp[i] += delta
                J[:, i] = (self.get_fk(qp) - p) / delta
            
            # Damped Least Squares 유사 (특이점 회피)
            step = np.linalg.pinv(J) @ err
            q += np.clip(step, -0.15, 0.15) 
            
            # [안전] 물리적 관절 한계 강제 적용 (Clip)
            q = np.clip(q, self.q_min, self.q_max)
        return q

    def control_loop(self):
        if self.current_state == self.STATE_NOT_READY: return
        
        now = time.time()
        cmd = Float64MultiArray()
        
        if self.current_state == self.STATE_FOLLOWING:
            # 부드러운 시작
            if self.is_first_follow_step:
                if now - self.last_valid_target_time < 0.1:
                    self.is_first_follow_step = False
                else:
                    self.target_q = self.current_joints
            
            # 타임아웃 체크
            time_since_last_target = now - self.last_valid_target_time
            if not self.is_first_follow_step and (time_since_last_target > 2.0):
                self.current_state = self.STATE_ESTOP
                self.get_logger().error(f"⚠️ Tracking Lost! (No data for {time_since_last_target:.1f}s)")
            else:
                self.target_q = self.numerical_ik(self.target_pose, self.current_joints)
                
        elif self.current_state == self.STATE_RESETTING:
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = self.STATE_IDLE
            else: self.target_q = self.current_joints + diff * 0.05
        elif self.current_state in [self.STATE_IDLE, self.STATE_ESTOP]:
            pass 
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralController()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
