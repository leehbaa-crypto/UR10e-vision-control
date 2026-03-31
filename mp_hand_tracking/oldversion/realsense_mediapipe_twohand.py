import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HandGestureNode(Node):
    def __init__(self):
        super().__init__('nrs_hand_gesture_node')
        
        # 1. ROS 2 퍼블리셔: FSM이 읽을 수 있는 형태로 메시지 발행
        self.gesture_pub = self.create_publisher(String, '/hand_gesture', 10)
        self.get_logger().info("Gesture Recognition Node Initialized.")

        # 2. MediaPipe 초기화 (양손 동시 인식)
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # 3. RealSense L515 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense L515 Camera Started.")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense: {e}")
            sys.exit(1)

        # ROS 2 루프 타이머 (약 30Hz)
        self.timer = self.create_timer(0.033, self.timer_callback)

    def recognize_gesture(self, hand_landmarks, handedness_label):
        """관절의 상대적 위치를 기반으로 5가지 제스처 판별"""
        tips = [
            self.mp_hands.HandLandmark.THUMB_TIP,
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        pips = [
            self.mp_hands.HandLandmark.THUMB_IP,
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]

        fingers_up = []
        
        # 엄지 판별 (x좌표 기준, 좌우 손 방향 고려)
        thumb_tip_x = hand_landmarks.landmark[tips[0]].x
        thumb_ip_x = hand_landmarks.landmark[pips[0]].x
        if handedness_label == 'Right':
            fingers_up.append(thumb_tip_x < thumb_ip_x)
        else:
            fingers_up.append(thumb_tip_x > thumb_ip_x)

        # 나머지 손가락 판별 (y좌표 기준: 끝이 마디보다 위에 있는지)
        for i in range(1, 5):
            tip_y = hand_landmarks.landmark[tips[i]].y
            pip_y = hand_landmarks.landmark[pips[i]].y
            fingers_up.append(tip_y < pip_y)

        # 엄지-검지 핀치(Okay) 판별용
        idx_tip = hand_landmarks.landmark[tips[1]]
        thm_tip = hand_landmarks.landmark[tips[0]]
        pinch_dist = np.sqrt((idx_tip.x - thm_tip.x)**2 + (idx_tip.y - thm_tip.y)**2)
        is_pinched = pinch_dist < 0.05

        # 5가지 상태 매칭
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
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        # 이미지 반전 (거울 모드)
        image = np.asanyarray(color_frame.get_data())
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(image_rgb)
        
        left_label = ""
        right_label = ""

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # MediaPipe가 판별한 라벨을 그대로 사용
                actual_hand = results.multi_handedness[idx].classification[0].label
                
                gesture = self.recognize_gesture(hand_landmarks, actual_hand)
                
                if actual_hand == "Left":
                    left_label = gesture
                else:
                    right_label = gesture
                    
                # 손목 위치에 제스처 텍스트 출력
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                wx, wy = int(wrist.x * 640), int(wrist.y * 480)
                cv2.putText(image, f"{actual_hand}: {gesture}", (wx - 20, wy + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. 선행 연구 FSM이 파싱할 수 있는 정확한 포맷으로 메시지 발행
        # ex) "Left:paper,Right:rock"
        msg = String()
        msg.data = f"Left:{left_label},Right:{right_label}"
        self.gesture_pub.publish(msg)

        # 화면 상단 대시보드 출력
        cv2.rectangle(image, (0, 0), (640, 50), (0, 0, 0), -1)
        cv2.putText(image, f"PUBLISHING -> {msg.data}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('NRS ROS 2 Gesture Recognition', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        self.pipeline.stop()
        self.hands.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandGestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == '__main__':
    main()
