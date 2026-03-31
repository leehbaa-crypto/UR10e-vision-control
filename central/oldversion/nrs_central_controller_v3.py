import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time
import trimesh
import os

class CentralControllerV3(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v3')
        
        self.current_state = "NOT READY"
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        # 필터 및 목표
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.target_rot = np.eye(3) # 6자유도 자세 목표
        self.alpha = 0.3 
        
        # 작업 범위 (로봇 베이스 기준)
        self.surface_bounds = {'x': [0.4, 0.7], 'y': [-0.3, 0.3]} 
        
        # DH Params (UR10e)
        self.dh = [
            {'a': 0,        'd': 0.1807,  'alpha': 1.570796},
            {'a': -0.6127,  'd': 0,       'alpha': 0},
            {'a': -0.57155, 'd': 0,       'alpha': 0},
            {'a': 0,        'd': 0.17415, 'alpha': 1.570796},
            {'a': 0,        'd': 0.11985, 'alpha': -1.570796},
            {'a': 0,        'd': 0.11655, 'alpha': 0}
        ]
        
        self.mesh = None
        self.gesture_history = [] # 슬라이딩 윈도우 투표용

        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(Point, '/guided_target', self.guided_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)
        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            try:
                # Trimesh로 STL 로드 (경로는 실제 환경에 맞게 수정 필요)
                self.mesh = trimesh.load(f"visual/{filename}")
                self.get_logger().info(f"✅ Mesh loaded for Guide Mode: {filename}")
                self.current_state = "IDLE"
            except Exception as e:
                self.get_logger().error(f"Failed to load mesh: {e}")

    def target_cb(self, msg):
        if self.current_state == "FOLLOWING":
            raw_target = np.array([msg.z + 0.2, -msg.x, -msg.y + 0.8])
            self.target_pose = self.alpha * raw_target + (1.0 - self.alpha) * self.target_pose
            self.target_rot = np.array([[0,0,-1], [0,1,0], [1,0,0]]) # 기본 자세 (아래 보기)

    def guided_cb(self, msg):
        if self.current_state == "GUIDED_FOLLOWING" and self.mesh is not None:
            # 2D 패드 비율(0~1)을 로봇의 물리적 작업 공간(X, Y)으로 매핑
            rx = self.surface_bounds['x'][0] + msg.y * (self.surface_bounds['x'][1] - self.surface_bounds['x'][0])
            ry = self.surface_bounds['y'][0] + (1.0 - msg.x) * (self.surface_bounds['y'][1] - self.surface_bounds['y'][0])
            
            # Raycasting으로 표면 높이와 법선 벡터 계산
            ray_origins = np.array([[rx, ry, 2.0]])
            ray_dirs = np.array([[0.0, 0.0, -1.0]])
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_dirs)
            
            if len(locations) > 0:
                # 매핑된 표면 좌표 적용
                surface_z = locations[0][2]
                self.target_pose = self.alpha * np.array([rx, ry, surface_z]) + (1.0 - self.alpha) * self.target_pose
                
                # 수직 자세(Normal Vector) 계산
                normal = self.mesh.face_normals[index_tri[0]]
                # 로봇 엔드이펙터 Z축이 표면 안쪽(-Normal)을 향하게 설정
                Z_tool = -normal
                Y_tool = np.array([0, 1, 0]) # 임의의 Y축 (Y 방향 유지)
                X_tool = np.cross(Y_tool, Z_tool)
                X_tool /= np.linalg.norm(X_tool)
                Y_tool = np.cross(Z_tool, X_tool)
                
                self.target_rot = np.column_stack((X_tool, Y_tool, Z_tool))

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        parts = msg.data.split(',')
        r, l = parts[0].split(':')[1], parts[1].split(':')[1]
        
        # 제스처 딜레이 해결 (투표 시스템)
        self.gesture_history.append((l, r))
        if len(self.gesture_history) > 10: self.gesture_history.pop(0)
        
        if len(self.gesture_history) == 10:
            count = { "paper_rock": 0, "paper_pointing": 0, "rock_rock": 0, "paper_paper": 0 }
            for h in self.gesture_history:
                if h == ("paper", "rock"): count["paper_rock"] += 1
                elif h == ("paper", "pointing"): count["paper_pointing"] += 1
                elif h == ("rock", "rock"): count["rock_rock"] += 1
                elif h == ("paper", "paper"): count["paper_paper"] += 1
            
            # 10프레임(약 0.3초) 중 7번 이상 일치하면 즉시 상태 전환
            if count["rock_rock"] >= 7: self.current_state = "ESTOP"
            elif count["paper_rock"] >= 7 and self.current_state in ["IDLE"]: self.current_state = "FOLLOWING"
            elif count["paper_pointing"] >= 7 and self.current_state in ["IDLE", "FOLLOWING"]: self.current_state = "GUIDED_FOLLOWING"
            elif count["paper_paper"] >= 7 and self.current_state not in ["IDLE", "RESETTING"]: self.current_state = "RESETTING"

    def get_fk_6dof(self, q):
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = self.dh[i]['a'], self.dh[i]['d'], self.dh[i]['alpha']
            ct, st = np.cos(q[i]), np.sin(q[i]); ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
            T = T @ Ti
        return T[:3, 3], T[:3, :3] # Position, Rotation Matrix

    def numerical_ik_6dof_dls(self, target_p, target_R, current_q):
        """ 6-DOF 수직 연마를 위한 확장된 역기구학 """
        q = current_q.copy()
        lambda_sq = 0.05 ** 2
        
        for _ in range(10):
            p, R = self.get_fk_6dof(q)
            
            # 1. 위치 오차
            err_p = target_p - p
            
            # 2. 자세 오차 (Orientation Error)
            err_o = 0.5 * (np.cross(R[:,0], target_R[:,0]) + np.cross(R[:,1], target_R[:,1]) + np.cross(R[:,2], target_R[:,2]))
            
            err = np.hstack((err_p, err_o))
            if np.linalg.norm(err) < 1e-3: break
            
            # 6x6 자코비안 계산
            J = np.zeros((6, 6))
            delta = 1e-5
            for i in range(6):
                qp = q.copy(); qp[i] += delta
                pp, Rp = self.get_fk_6dof(qp)
                J[:3, i] = (pp - p) / delta
                err_o_p = 0.5 * (np.cross(R[:,0], Rp[:,0]) + np.cross(R[:,1], Rp[:,1]) + np.cross(R[:,2], Rp[:,2]))
                J[3:, i] = err_o_p / delta
                
            # DLS
            JJt = J @ J.T
            J_dls = J.T @ np.linalg.inv(JJt + lambda_sq * np.eye(6))
            
            step = J_dls @ err
            q += np.clip(step, -0.1, 0.1)
        return q

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()
        
        if self.current_state in ["FOLLOWING", "GUIDED_FOLLOWING"]:
            self.target_q = self.numerical_ik_6dof_dls(self.target_pose, self.target_rot, self.current_joints)
            
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralControllerV3()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
