import sys
import os
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point

class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('nrs_hand_tracking_node')
        
        # 1. ROS 2 퍼블리셔(Publisher) 생성
        # - 제스처 상태("pinch", "open_hand")를 FSM 노드로 전송
        self.gesture_pub = self.create_publisher(String, '/hand_gesture', 10)
        # - 3D 좌표(X, Y, Z)를 추후 로봇 제어기나 역기구학 노드로 전송
        self.target_pub = self.create_publisher(Point, '/target_pose', 10)
        
        self.get_logger().info("Hand Tracking Node Initialized. Starting RealSense...")

        # 2. MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=0, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

        # 3. RealSense L515 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.intr = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.get_logger().info("RealSense L515 Connected successfully!")
        except Exception as e:
            self.get_logger().error(f"Error starting RealSense: {e}")
            sys.exit(1)

        # 제어용 변수
        self.ema_alpha = 0.3
        self.filtered_point = None
        self.pinch_threshold = 0.04
        
        # 4. ROS 2 타이머 설정 (기존 while True 루프를 대체)
        # 약 30FPS로 동작하도록 0.033초마다 콜백 함수 실행
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        # 카메라 프레임 읽기 및 정렬
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame: 
            return

        image = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 추론
        results = self.hands.process(image_rgb)
        
        # 기본 제스처 상태는 손을 편 상태로 정의
        current_gesture = "open_hand"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                
                ix, iy = int(index_tip.x * 640), int(index_tip.y * 480)
                tx, ty = int(thumb_tip.x * 640), int(thumb_tip.y * 480)
                
                if 0 <= ix < 640 and 0 <= iy < 480 and 0 <= tx < 640 and 0 <= ty < 480:
                    dist_index = depth_frame.get_distance(ix, iy)
                    dist_thumb = depth_frame.get_distance(tx, ty)
                    
                    if dist_index > 0 and dist_thumb > 0:
                        p_index = rs.rs2_deproject_pixel_to_point(self.intr, [ix, iy], dist_index)
                        p_thumb = rs.rs2_deproject_pixel_to_point(self.intr, [tx, ty], dist_thumb)
                        
                        # 거리 계산 및 핀치(클러치) 상태 판별
                        pinch_dist = np.linalg.norm(np.array(p_index) - np.array(p_thumb))
                        if pinch_dist < self.pinch_threshold:
                            current_gesture = "pinch"
                            
                        # EMA 필터 적용 (노이즈 제거)
                        raw_point = np.array(p_index)
                        if self.filtered_point is None:
                            self.filtered_point = raw_point
                        else:
                            self.filtered_point = self.ema_alpha * raw_point + (1 - self.ema_alpha) * self.filtered_point
                            
                        fx, fy, fz = self.filtered_point
                        
                        # --- 데이터 퍼블리시 (3D 좌표) ---
                        msg_point = Point()
                        msg_point.x, msg_point.y, msg_point.z = float(fx), float(fy), float(fz)
                        self.target_pub.publish(msg_point)
                        
                        # 화면 시각화 업데이트
                        color = (0, 0, 255) if current_gesture == "pinch" else (255, 0, 0)
                        cv2.circle(image, (ix, iy), 12, color, -1)
                        cv2.putText(image, f"3D Target: {fx:.3f}, {fy:.3f}, {fz:.3f} m", 
                                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- 데이터 퍼블리시 (제스처 상태) ---
        msg_gesture = String()
        msg_gesture.data = current_gesture
        self.gesture_pub.publish(msg_gesture)
        
        cv2.putText(image, f"Published Gesture: {current_gesture}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ROS 2 MediaPipe + L515 Tracking', image)
        
        # 화면 종료 키 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def destroy_node(self):
        # 노드 종료 시 카메라 파이프라인 안전하게 닫기
        self.pipeline.stop()
        self.hands.close()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    
    try:
        # ROS 2 이벤트 루프 실행 (타이머 콜백이 계속 호출됨)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy.shutdown() 은 위에서 처리되거나 스크립트 종료 시 알아서 정리됨

if __name__ == '__main__':
    main()
