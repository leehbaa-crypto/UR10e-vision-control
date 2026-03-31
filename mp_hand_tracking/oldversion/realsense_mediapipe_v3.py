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

class UnifiedHandControlNode(Node):
    def __init__(self):
        super().__init__('nrs_unified_hand_control_node')
        
        # 1. ROS 2 퍼블리셔 설정
        self.gesture_pub = self.create_publisher(String, '/hand_gesture', 10)
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)
        
        self.get_logger().info("Unified Hand Control Node V3 Initialized.")

        # 2. MediaPipe 초기화 (양손 동시 인식)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # 3. RealSense L515 파이프라인 설정 (Color + Depth 동시 활성화)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.get_logger().info("RealSense L515 Camera Started successfully!")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense: {e}")
            sys.exit(1)

        # 3D 궤적 제어용 필터 변수
        self.ema_alpha = 0.3
        self.filtered_point = None

        # 루프 타이머 (약 30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)

    def recognize_gesture(self, hand_landmarks, handedness_label):
        """관절의 상대적 위치를 기반으로 5가지 제스처 판별"""
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        fingers_up = []
        
        # 엄지 판별
        thumb_tip_x = hand_landmarks.landmark[tips[0]].x
        thumb_ip_x = hand_landmarks.landmark[pips[0]].x
        if handedness_label == 'Right':
            fingers_up.append(thumb_tip_x < thumb_ip_x)
        else:
            fingers_up.append(thumb_tip_x > thumb_ip_x)

        # 나머지 손가락 판별
        for i in range(1, 5):
            fingers_up.append(hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[pips[i]].y)

        # 핀치(Okay) 거리 판별
        idx_tip = hand_landmarks.landmark[tips[1]]
        thm_tip = hand_landmarks.landmark[tips[0]]
        pinch_dist = np.sqrt((idx_tip.x - thm_tip.x)**2 + (idx_tip.y - thm_tip.y)**2)
        is_pinched = pinch_dist < 0.05

        if is_pinched and fingers_up[2] and fingers_up[3] and fingers_up[4]:
            return "okay"
        elif fingers_up.count(True) >= 4:
            return "paper"
        elif fingers_up.count(True) <= 1 and not fingers_up[1]:
            return "rock"
        elif fingers_up[1] and fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            return "scissors"
        elif fingers_up[1] and not fingers_up[2] and not fingers_up[3] and not fingers_up[4]:
            return "pointing"
        else:
            return ""

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return

        # 이미지 반전 (거울 모드 UI를 위해)
        image = np.asanyarray(color_frame.get_data())
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(image_rgb)
        
        left_label = ""
        right_label = ""
        target_3d_str = "No Target"

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 반전된 이미지이므로 MediaPipe의 라벨을 그대로 사용
                actual_hand = results.multi_handedness[idx].classification[0].label
                gesture = self.recognize_gesture(hand_landmarks, actual_hand)
                
                # --- [왼손 로직] FSM 제어용 제스처 인식 ---
                if actual_hand == "Left":
                    left_label = gesture
                    
                # --- [오른손 로직] 3D 궤적 추종 및 제스처 인식 ---
                else:
                    right_label = gesture
                    
                    # 1. 검지 끝 좌표 추출 (이미지 해상도 기준)
                    index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy = int(index_tip.x * 640), int(index_tip.y * 480)
                    
                    # 2. 핵심 트릭: 이미지를 좌우 반전시켰으므로, 원본 뎁스 맵에 접근하려면 X 좌표를 다시 뒤집어야 함
                    depth_x = 640 - ix - 1
                    
                    if 0 <= depth_x < 640 and 0 <= iy < 480:
                        dist = depth_frame.get_distance(depth_x, iy)
                        
                        if dist > 0: # 뎁스 노이즈 방어
                            # 3. 2D 픽셀을 현실의 3D 공간(m) 좌표로 변환
                            p_index = rs.rs2_deproject_pixel_to_point(self.intr, [depth_x, iy], dist)
                            raw_point = np.array(p_index)
                            
                            # 4. EMA 필터 적용 (로봇 떨림 방지)
                            if self.filtered_point is None:
                                self.filtered_point = raw_point
                            else:
                                self.filtered_point = self.ema_alpha * raw_point + (1 - self.ema_alpha) * self.filtered_point
                                
                            fx, fy, fz = self.filtered_point
                            
                            # 5. 3D 좌표 퍼블리시
                            msg_point = Point()
                            msg_point.x, msg_point.y, msg_point.z = float(fx), float(fy), float(fz)
                            self.target_pub.publish(msg_point)
                            
                            target_3d_str = f"{fx:.3f}, {fy:.3f}, {fz:.3f} m"
                            
                            # 시각화: 오른손 검지 끝에 빨간 원 그리기
                            cv2.circle(image, (ix, iy), 10, (0, 0, 255), -1)

                # 손목에 제스처 텍스트 출력
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                wx, wy = int(wrist.x * 640), int(wrist.y * 480)
                cv2.putText(image, f"{actual_hand}: {gesture}", (wx - 20, wy + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # FSM용 제스처 토픽 발행
        msg_gesture = String()
        msg_gesture.data = f"Left:{left_label},Right:{right_label}"
        self.gesture_pub.publish(msg_gesture)

        # 화면 UI 출력
        cv2.rectangle(image, (0, 0), (640, 80), (0, 0, 0), -1)
        cv2.putText(image, f"FSM CMD: {msg_gesture.data}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, f"TARGET (Right): {target_3d_str}", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 255), 2)

        cv2.imshow('Unified Hand Control V3', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        self.pipeline.stop()
        self.hands.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedHandControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
