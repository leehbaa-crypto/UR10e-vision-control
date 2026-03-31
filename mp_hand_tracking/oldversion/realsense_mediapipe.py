import sys
import os

# 현재 디렉토리를 경로에서 일시적으로 제거하여 라이브러리 충돌 방지
current_dir = os.getcwd()
if current_dir in sys.path:
    sys.path.remove(current_dir)

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import time

def main():
    # 1. MediaPipe Hand 초기화
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        model_complexity=0, # 0: Lite (가장 빠름), 1: Full
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. RealSense L515 설정
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 컬러 및 깊이 스트림 활성화
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        profile = pipeline.start(config)
        print("RealSense L515 3D Hand Tracking Started!")
        
        # 깊이-컬러 정렬 도구 생성 (중요: 2D 픽셀을 3D 좌표로 바꿀 때 필요)
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # 카메라 내부 파라미터(Intrinsics) 가져오기
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
    except Exception as e:
        print(f"Error starting RealSense: {e}")
        return

    print("\n>>> 3D Tracking Started! Press 'q' to quit.")
    
    prev_time = 0

    try:
        while True:
            # 프레임 정렬 (컬러 이미지에 깊이 데이터를 맞춤)
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame: continue

            # 이미지 변환
            image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # MediaPipe 추론
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 모든 랜드마크 그리기
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 검지 끝(Index Finger Tip, Landmark 8)의 3D 좌표 추출
                    # MediaPipe 좌표(0~1)를 픽셀 좌표로 변환
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix = int(index_tip.x * 640)
                    iy = int(index_tip.y * 480)
                    
                    # 범위 체크 (이미지 밖으로 나가는 경우 방지)
                    if 0 <= ix < 640 and 0 <= iy < 480:
                        # 깊이 값 가져오기 (단위: mm -> m 변환 위해 0.001 곱함)
                        distance = depth_frame.get_distance(ix, iy)
                        
                        # 2D 픽셀 + 깊이 데이터를 사용하여 실제 3D 좌표(X, Y, Z) 계산 (단위: meter)
                        point = rs.rs2_deproject_pixel_to_point(intr, [ix, iy], distance)
                        tx, ty, tz = point # tx: 가로, ty: 세로, tz: 깊이(앞뒤)
                        
                        # 화면에 3D 좌표 출력 (미터 단위)
                        cv2.putText(image, f"3D Index Tip: {tx:.2f}, {ty:.2f}, {tz:.2f} (m)", 
                                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # 검지 끝부분에 원 그리기
                        cv2.circle(image, (ix, iy), 10, (255, 0, 0), -1)

            # 정보 표시
            cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('MediaPipe 3D Hand Tracking (L515)', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        hands.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
