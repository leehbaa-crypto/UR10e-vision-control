import sys
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point
import threading
import time

class UnifiedVisionNodeLightV7(Node):
    def __init__(self):
        super().__init__('nrs_vision_node_light_v7')
        
        self.gesture_pub = self.create_publisher(String, '/hand_gesture', 10)
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)
        self.guided_pub = self.create_publisher(Point, '/guided_target', 10)
        self.surface_cmd_pub = self.create_publisher(String, '/surface_command', 10)
        
        self.robot_state = "NOT READY"
        self.create_subscription(String, '/robot_status', self.status_cb, 10)
        
        self.surfaces = ["flat_surface_5.stl", "concave_surface_1.stl", "concave_surface_2.stl", 
                         "_concave_surface_0.75.stl", "_comp_concave_0_75_v0_42.stl", "compound_concave_0_75_v0_42.stl"]
        self.current_idx = 0
        self.launch_triggered = False
        self.last_left_gesture = "None"

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

        self.is_running = True
        self.vision_thread = threading.Thread(target=self.camera_loop)
        self.vision_thread.start()
        self.get_logger().info("✅ LIGHT Vision V7 Started (Scissors Gesture Added)")

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
        # 💡 [디버깅용 추가] 검지와 중지만 폈을 때 "가위(scissors)"로 인식
        if up[1] and up[2] and not up[3] and not up[4]: return "scissors"
        if up.count(True) >= 4: return "paper"
        if up.count(True) <= 1 and not up[1]: return "rock"
        if up[1] and not up[2]: return "pointing"
        return "None"

    def camera_loop(self):
        pad_rect = (120, 150, 520, 350) 
        prev_time = time.time()
        
        while self.is_running and rclpy.ok():
            curr_time = time.time()
            if curr_time - prev_time < 0.05:
                time.sleep(0.01); continue
            prev_time = curr_time

            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_frame or not depth_frame: continue

            image = np.asanyarray(color_frame.get_data())
            small_img = cv2.resize(image, (320, 240))
            results = self.hands.process(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
            
            gestures = {"Left": "None", "Right": "None"}
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label 
                    gesture = self.recognize_gesture(hand_landmarks, label)
                    
                    if label == "Right": 
                        gestures["Left"] = gesture 
                        if gesture == "pointing" and self.last_left_gesture != "pointing" and self.robot_state == "NOT READY":
                            self.current_idx = (self.current_idx + 1) % len(self.surfaces)
                        self.last_left_gesture = gesture
                        
                    elif label == "Left": 
                        gestures["Right"] = gesture 
                        cx, cy = int(hand_landmarks.landmark[8].x * 640), int(hand_landmarks.landmark[8].y * 480)

                        if self.robot_state == "FOLLOWING":
                            if 0 <= cx < 640 and 0 <= cy < 480:
                                d_list = []
                                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (0,0)]: 
                                    val = depth_frame.get_distance(cx+dx, cy+dy)
                                    if val > 0.05: d_list.append(val)
                                if len(d_list) > 0:
                                    d = np.median(d_list)
                                    p3d = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], d)
                                    self.target_pub.publish(Point(x=float(p3d[0]), y=float(p3d[1]), z=float(p3d[2])))
                                    cv2.circle(image, (cx, cy), 15, (0, 255, 0), -1)

                        elif self.robot_state == "GUIDED_FOLLOWING":
                            cv2.circle(image, (cx, cy), 10, (255, 0, 255), -1)
                            px = np.clip((cx - pad_rect[0]) / (pad_rect[2] - pad_rect[0]), 0.0, 1.0)
                            py = np.clip((cy - pad_rect[1]) / (pad_rect[3] - pad_rect[1]), 0.0, 1.0)
                            self.guided_pub.publish(Point(x=float(px), y=float(py), z=0.0))

                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            self.gesture_pub.publish(String(data=f"Left:{gestures['Left']},Right:{gestures['Right']}"))

            if gestures["Left"] == "okay" and gestures["Right"] == "okay":
                if not self.launch_triggered and self.robot_state == "NOT READY":
                    msg = String(); msg.data = f"LAUNCH:{self.surfaces[self.current_idx]}"
                    self.surface_cmd_pub.publish(msg)
                    self.launch_triggered = True
            elif self.robot_state != "NOT READY": 
                self.launch_triggered = True 
            else: 
                self.launch_triggered = False

            display = cv2.flip(image, 1)

            cv2.rectangle(display, (0, 0), (640, 100), (0, 0, 0), -1)
            status_color = (0, 255, 0) if self.robot_state == "FOLLOWING" else (255, 100, 255) if self.robot_state == "GUIDED_FOLLOWING" else (0, 165, 255) if self.robot_state != "ESTOP" else (0, 0, 255)
            cv2.putText(display, f"ROBOT: {self.robot_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display, f"FPS: {1.0/(time.time()-curr_time+0.001):.1f}", (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.robot_state == "NOT READY":
                cv2.putText(display, f"R-POINT to CHANGE: {self.surfaces[self.current_idx]}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            flipped_pad = (640 - pad_rect[2], pad_rect[1], 640 - pad_rect[0], pad_rect[3])
            if self.robot_state == "GUIDED_FOLLOWING":
                cv2.rectangle(display, (flipped_pad[0], flipped_pad[1]), (flipped_pad[2], flipped_pad[3]), (255, 0, 255), 2)
                cv2.putText(display, f"PAD: {self.surfaces[self.current_idx]}", (flipped_pad[0], flipped_pad[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv2.imshow('Vision V7', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    def destroy_node(self):
        self.is_running = False
        self.vision_thread.join()
        self.pipeline.stop(); cv2.destroyAllWindows(); super().destroy_node()

def main(args=None):
    rclpy.init(args=args); node = UnifiedVisionNodeLightV7()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__': main()
