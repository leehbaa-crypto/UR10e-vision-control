import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState

class NRSUnifiedCore(Node):
    def __init__(self):
        super().__init__('nrs_unified_core')
        
        # 1. 상태 및 제어 변수 (Controller 기능 통합)
        self.STATE_NOT_READY = "NOT READY"
        self.STATE_IDLE = "IDLE"
        self.STATE_FOLLOWING = "FOLLOWING"
        self.STATE_ESTOP = "ESTOP"
        self.STATE_RESETTING = "RESETTING"
        self.current_state = self.STATE_NOT_READY
        
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        self.target_pose = np.array([0.5, 0, 0.5])
        self.last_valid_target_time = time.time()
        
        # UR10e DH Params
        self.dh_params = [
            {'a': 0,        'd': 0.1807,  'alpha': 1.570796},
            {'a': -0.6127,  'd': 0,       'alpha': 0},
            {'a': -0.57155, 'd': 0,       'alpha': 0},
            {'a': 0,        'd': 0.17415, 'alpha': 1.570796},
            {'a': 0,        'd': 0.11985, 'alpha': -1.570796},
            {'a': 0,        'd': 0.11655, 'alpha': 0}
        ]

        # 2. ROS 2 통신 설정
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.surface_pub = self.create_publisher(String, '/surface_command', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # 3. 비전 설정 (경량화)
        self.surfaces = ["flat_surface_5.stl", "concave_surface_1.stl", "concave_surface_2.stl", 
                         "_concave_surface_0.75.stl", "_comp_concave_0_75_v0_42.stl", "compound_concave_0_75_v0_42.stl"]
        self.current_idx = 0
        self.last_left_gesture = ""
        self.last_right_gesture = ""
        self.gesture_start_time = None
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) # 정밀도 소폭 조정
        self.mp_drawing = mp.solutions.drawing_utils

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except Exception as e:
            self.get_logger().error(f"RealSense Error: {e}"); sys.exit(1)

        # 주기: 20Hz (50ms) - 경량화 포인트
        self.timer = self.create_timer(0.05, self.core_loop)
        self.get_logger().info("🚀 NRS Unified Core Started (Optimized for performance)")

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def recognize_gesture(self, hand_landmarks, label):
        tips = [4, 8, 12, 16, 20]; pips = [3, 6, 10, 14, 18]; up = []
        if label == 'Right': up.append(hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x)
        else: up.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)
        for i in range(1, 5): up.append(hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y)
        
        dist = np.linalg.norm(np.array([hand_landmarks.landmark[8].x - hand_landmarks.landmark[4].x, 
                                        hand_landmarks.landmark[8].y - hand_landmarks.landmark[4].y]))
        if dist < 0.08 and up[2] and up[3] and up[4]: return "okay"
        if up.count(True) >= 4: return "paper"
        if up.count(True) <= 1 and not up[1]: return "rock"
        if up[1] and not up[2]: return "pointing"
        return "None"

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
        for _ in range(15): # 반복 횟수 최적화
            p = self.get_fk(q); err = target - p
            if np.linalg.norm(err) < 1e-3: break
            J = np.zeros((3, 6)); delta = 1e-6
            for i in range(6):
                qp = q.copy(); qp[i] += delta
                J[:, i] = (self.get_fk(qp) - p) / delta
            q += np.clip(np.linalg.pinv(J) @ err, -0.06, 0.06)
        return q

    def core_loop(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame: return

        img = np.asanyarray(color_frame.get_data())
        # 연산용 이미지 축소 (경량화 포인트)
        small_img = cv2.resize(img, (320, 240))
        results = self.hands.process(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
        
        gestures = {"Left": "None", "Right": "None"}
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                gesture = self.recognize_gesture(hand_landmarks, label)
                gestures[label] = gesture
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS) # 원본에 그리기

                if label == "Right": # 사용자 왼손
                    if gesture == "pointing" and self.last_right_gesture != "pointing" and self.current_state == self.STATE_NOT_READY:
                        self.current_idx = (self.current_idx + 1) % len(self.surfaces)
                    self.last_right_gesture = gesture
                elif label == "Left": # 사용자 오른손
                    tip = hand_landmarks.landmark[8]
                    cx, cy = int(tip.x * 640), int(tip.y * 480)
                    if 0 <= cx < 640 and 0 <= cy < 480:
                        d = depth_frame.get_distance(cx, cy)
                        if d > 0:
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], d)
                            self.target_pose = np.array([p3d[2] + 0.2, -p3d[0], -p3d[1] + 0.8])
                            self.last_valid_target_time = time.time()

        # FSM 및 IK 처리
        now = time.time()
        l, r = gestures["Left"], gestures["Right"] # Left:사용자오른손, Right:사용자왼손
        
        if r == "okay" and l == "okay":
            if self.current_state == self.STATE_NOT_READY:
                msg = String(); msg.data = f"LAUNCH:{self.surfaces[self.current_idx]}"
                self.surface_pub.publish(msg)
                self.current_state = self.STATE_IDLE
        
        if l == "rock" and r == "rock" and self.current_state != self.STATE_NOT_READY:
            self.current_state = self.STATE_ESTOP
        
        if r == "paper" and l == "rock" and self.current_state == self.STATE_IDLE:
            if self.gesture_start_time is None: self.gesture_start_time = now
            elif now - self.gesture_start_time >= 2.0: # 2초로 단축
                self.current_state = self.STATE_FOLLOWING
                self.gesture_start_time = None
        elif r == "paper" and l == "paper" and self.current_state in [self.STATE_IDLE, self.STATE_ESTOP]:
            if self.gesture_start_time is None: self.gesture_start_time = now
            elif now - self.gesture_start_time >= 2.0:
                self.current_state = self.STATE_RESETTING
                self.gesture_start_time = None
        else: self.gesture_start_time = None

        # 명령 발행
        cmd = Float64MultiArray()
        if self.current_state == self.STATE_FOLLOWING:
            if now - self.last_valid_target_time > 1.0: self.current_state = self.STATE_ESTOP
            else: self.target_q = self.numerical_ik(self.target_pose, self.current_joints)
        elif self.current_state == self.STATE_RESETTING:
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = self.STATE_IDLE
            else: self.target_q = self.current_joints + diff * 0.1
        elif self.current_state == self.STATE_IDLE:
            self.target_q = self.home_joints
        
        if self.current_state != self.STATE_NOT_READY:
            cmd.data = self.target_q.tolist()
            self.cmd_pub.publish(cmd)

        # UI 출력
        disp = cv2.flip(img, 1)
        cv2.rectangle(disp, (0,0), (640, 100), (0,0,0), -1)
        color = (0,255,0) if self.current_state == "FOLLOWING" else (0,165,255)
        cv2.putText(disp, f"STATE: {self.current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if self.current_state == "NOT READY":
            cv2.putText(disp, f"SELECT: {self.surfaces[self.current_idx]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.imshow('Unified Control Core', disp)
        if cv2.waitKey(1) & 0xFF == ord('q'): rclpy.shutdown()

def main():
    rclpy.init(); node = NRSUnifiedCore()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
