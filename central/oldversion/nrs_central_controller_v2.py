import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time

class CentralControllerV2(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v2')
        
        # 1. 상태 정의 (GUIDED_FOLLOWING 추가)
        self.STATE_NOT_READY = "NOT READY"
        self.STATE_IDLE = "IDLE"
        self.STATE_FOLLOWING = "FOLLOWING"
        self.STATE_GUIDED_FOLLOWING = "GUIDED_FOLLOWING" # 새 기능용 상태
        self.STATE_ESTOP = "ESTOP"
        self.STATE_RESETTING = "RESETTING"
        self.current_state = self.STATE_NOT_READY
        
        # 2. 제어 변수
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        self.target_pose = np.array([0.5, 0.0, 0.5])
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

        # 20Hz (0.05s) 주기
        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.get_logger().info(f"🚀 Controller V2 (with Guided Mode Support) Ready.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            self.current_state = self.STATE_IDLE
            self.get_logger().info("✅ Simulation IDLE")

    def target_cb(self, msg):
        # Vision 좌표 -> 로봇 좌표계 변환 (기존 로직 유지)
        self.target_pose = np.array([msg.z + 0.2, -msg.x, -msg.y + 0.8])
        self.last_valid_target_time = time.time()

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        try:
            parts = msg.data.split(',')
            self.current_gestures["User_Right"] = parts[0].split(':')[1]
            self.current_gestures["User_Left"] = parts[1].split(':')[1]
            
            if time.time() - self.last_log_time > 1.0:
                print(f"👋 [{self.current_state}] L:{self.current_gestures['User_Left']} R:{self.current_gestures['User_Right']}")
                self.last_log_time = time.time()
            self.update_state_machine()
        except: pass

    def update_state_machine(self):
        l, r = self.current_gestures["User_Left"], self.current_gestures["User_Right"]
        now = time.time()

        # 1. 비상 정지 (L:주먹, R:주먹)
        if l == "rock" and r == "rock":
            if self.current_state not in [self.STATE_ESTOP, self.STATE_NOT_READY]:
                self.current_state = self.STATE_ESTOP
                self.get_logger().warn("🚨 EMERGENCY STOP")
            return

        # 2. 일반 추종 모드 (L:보, R:주먹)
        if l == "paper" and r == "rock":
            if self.current_state in [self.STATE_IDLE, self.STATE_NOT_READY]:
                if self.gesture_start_time is None: self.gesture_start_time = now
                elif now - self.gesture_start_time >= self.required_duration:
                    self.current_state = self.STATE_FOLLOWING
                    self.last_valid_target_time = time.time()
                    self.is_first_follow_step = True
                    self.gesture_start_time = None
            else: self.gesture_start_time = None

        # 3. 가이드 추종 모드 (L:보, R:가리키기) - 새 기능
        elif l == "paper" and r == "pointing":
            if self.current_state in [self.STATE_IDLE, self.STATE_FOLLOWING]:
                if self.gesture_start_time is None: self.gesture_start_time = now
                elif now - self.gesture_start_time >= self.required_duration:
                    self.current_state = self.STATE_GUIDED_FOLLOWING
                    self.get_logger().info("🎯 Entering GUIDED FOLLOWING Mode")
                    self.last_valid_target_time = time.time()
                    self.is_first_follow_step = True
                    self.gesture_start_time = None
            else: self.gesture_start_time = None
        
        # 4. 리셋 (L:보, R:보)
        elif l == "paper" and r == "paper":
            if self.current_state in [self.STATE_IDLE, self.STATE_ESTOP, self.STATE_FOLLOWING, self.STATE_GUIDED_FOLLOWING]:
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
        for _ in range(7):
            p = self.get_fk(q); err = target - p
            if np.linalg.norm(err) < 1e-3: break
            J = np.zeros((3, 6)); delta = 1e-6
            for i in range(6):
                qp = q.copy(); qp[i] += delta
                J[:, i] = (self.get_fk(qp) - p) / delta
            q += np.clip(np.linalg.pinv(J) @ err, -0.05, 0.05)
        return q

    def control_loop(self):
        if self.current_state == self.STATE_NOT_READY: return
        
        now = time.time()
        cmd = Float64MultiArray()
        
        if self.current_state in [self.STATE_FOLLOWING, self.STATE_GUIDED_FOLLOWING]:
            # 부드러운 시작
            if self.is_first_follow_step:
                if now - self.last_valid_target_time < 0.1:
                    self.is_first_follow_step = False
                else:
                    self.target_q = self.current_joints
            
            # 타임아웃 체크 (2초 이상 데이터 없으면 정지)
            if not self.is_first_follow_step and (now - self.last_valid_target_time > 2.0):
                self.current_state = self.STATE_ESTOP
                self.get_logger().error("⚠️ Tracking Lost!")
            else:
                # [여기에 가이드 모드 전용 보정 로직을 추가할 수 있습니다]
                # 현재는 기본 추종과 동일하게 작동하며, 추후 self.target_pose를 커스터마이징 가능합니다.
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
    rclpy.init(); node = CentralControllerV2()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
