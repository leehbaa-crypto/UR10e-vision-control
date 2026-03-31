import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point

class UnifiedHandControlNodeV4(Node):
    def __init__(self):
        super().__init__('nrs_unified_hand_control_node_v4')
        
        self.gesture_pub = self.create_publisher(String, '/hand_gesture', 10)
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)
        self.surface_cmd_pub = self.create_publisher(String, '/surface_command', 10)
        
        self.robot_state = "NOT READY"
        self.create_subscription(String, '/robot_status', self.status_cb, 10)
        
        self.surfaces = ["flat_surface_5.stl", "concave_surface_1.stl", "concave_surface_2.stl", 
                         "_concave_surface_0.75.stl", "_comp_concave_0_75_v0_42.stl", "compound_concave_0_75_v0_42.stl"]
        self.current_idx = 0
        self.last_left_gesture = ""; self.last_right_gesture = ""
        self.launch_triggered = False

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)

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

        # 20FPS로 하향 조정 (경량화)
        self.timer = self.create_timer(0.05, self.timer_callback)

    def status_cb(self, msg):
        self.robot_state = msg.data

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

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: return

        image = np.asanyarray(color_frame.get_data())
        small_image = cv2.resize(image, (320, 240))
        results = self.hands.process(cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB))
        
        current_gestures = {"Left": "None", "Right": "None"}
        
        # [복구] 원본 이미지에서 연산 수행
        if results.multi_hand_landmarks:
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            if not depth_frame: return

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                gesture = self.recognize_gesture(hand_landmarks, label)
                current_gestures[label] = gesture
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if label == "Right": # 사용자 왼손
                    if gesture == "pointing" and self.last_right_gesture != "pointing" and self.robot_state == "NOT READY":
                        self.current_idx = (self.current_idx + 1) % len(self.surfaces)
                    self.last_right_gesture = gesture
                elif label == "Left": # 사용자 오른손
                    tip = hand_landmarks.landmark[8]
                    cx, cy = int(tip.x * 640), int(tip.y * 480)
                    if 0 <= cx < 640 and 0 <= cy < 480:
                        d_list = []
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                if 0 <= cx+dx < 640 and 0 <= cy+dy < 480:
                                    val = depth_frame.get_distance(cx+dx, cy+dy)
                                    if val > 0.1: d_list.append(val)
                        
                        if len(d_list) > 0:
                            d = np.median(d_list)
                            p3d = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], d)
                            target_msg = Point()
                            target_msg.x, target_msg.y, target_msg.z = float(p3d[0]), float(p3d[1]), float(p3d[2])
                            self.target_pub.publish(target_msg)
                            cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
                        else:
                            cv2.putText(image, "DEPTH LOST", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 화면 뒤집기 (거울 모드)
        display_image = cv2.flip(image, 1)
        
        gesture_msg = String()
        gesture_msg.data = f"Left:{current_gestures['Left']},Right:{current_gestures['Right']}"
        self.gesture_pub.publish(gesture_msg)

        if current_gestures["Left"] == "okay" and current_gestures["Right"] == "okay":
            if not self.launch_triggered and self.robot_state == "NOT READY":
                msg = String(); msg.data = f"LAUNCH:{self.surfaces[self.current_idx]}"
                self.surface_cmd_pub.publish(msg)
                self.launch_triggered = True
        elif self.robot_state != "NOT READY": self.launch_triggered = True 
        else: self.launch_triggered = False

        # [수정] 뒤집힌 화면 위에 글자를 다시 써서 똑바로 보이게 함
        cv2.rectangle(display_image, (0, 0), (640, 100), (0, 0, 0), -1)
        status_color = (0, 255, 0) if self.robot_state == "FOLLOWING" else (0, 165, 255)
        cv2.putText(display_image, f"ROBOT: {self.robot_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        if self.robot_state == "NOT READY":
            cv2.putText(display_image, f"L-POINT to CHANGE: {self.surfaces[self.current_idx]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Gesture Control Center (Lite)', display_image)

        if current_gestures["Left"] == "okay" and current_gestures["Right"] == "okay":
            if not self.launch_triggered and self.robot_state == "NOT READY":
                msg = String(); msg.data = f"LAUNCH:{self.surfaces[self.current_idx]}"
                self.surface_cmd_pub.publish(msg)
                self.launch_triggered = True
        elif self.robot_state != "NOT READY": self.launch_triggered = True 
        else: self.launch_triggered = False

        cv2.rectangle(display_image, (0, 0), (640, 100), (0, 0, 0), -1)
        status_color = (0, 255, 0) if self.robot_state == "FOLLOWING" else (0, 165, 255)
        cv2.putText(display_image, f"ROBOT: {self.robot_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        if self.robot_state == "NOT READY":
            cv2.putText(display_image, f"L-POINT to CHANGE: {self.surfaces[self.current_idx]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Gesture Control Center (Lite)', display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): rclpy.shutdown()

    def destroy_node(self):
        self.pipeline.stop(); self.hands.close(); cv2.destroyAllWindows(); super().destroy_node()

def main(args=None):
    rclpy.init(args=args); node = UnifiedHandControlNodeV4()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__': main()
