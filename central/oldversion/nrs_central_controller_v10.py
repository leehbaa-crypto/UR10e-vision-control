import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import math
import time
import trimesh
import os

class CentralControllerV9(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v9')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_state = "NOT READY"
        self.resume_state = "IDLE" 
        
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        # 💡 [핵심 해결 1] 로봇을 유도할 "가상의 당근(실시간 목표점)" 변수 추가
        self.current_ik_target_pos = np.array([0.5, 0.0, 0.5])
        self.current_ik_target_z_dir = np.array([0.0, 0.0, -1.0])
        self.needs_ik_reset = False # 모드 전환 시 순간이동 방지용 플래그
        
        # 사용자가 지시한 "최종 목적지"
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.target_z_dir = np.array([0.0, 0.0, -1.0]) 
        
        self.deadzone = 0.002   
        
        # DH 파라미터 및 툴 길이 (레퍼런스 코드 100% 일치)
        self.d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
        self.a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])
        self.alp = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
        self.tool_len = 0.27 
        self.tcp_z_offset = 0.0 
        
        self.q_min = np.array([-6.28]*6)
        self.q_max = np.array([ 6.28]*6)
        
        self.workspace = {'x': [0.35, 0.9], 'y': [-0.5, 0.5], 'z': [0.05, 0.6]}
        self.surface_bounds = {'x': [0.4, 0.7], 'y': [-0.3, 0.3]} 
        
        self.mesh = None
        self.gesture_history = [] 

        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(Point, '/guided_target', self.guided_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 20Hz 제어 루프
        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.last_valid_time = time.time()

        self.get_logger().info("🌟 Central Controller V9 (Anti-Teleportation & Perfect IK Reset) Ready.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            mesh_path = os.path.join(self.script_dir, "visual", filename)
            try:
                mesh = trimesh.load(mesh_path)
                mesh.apply_translation([0.5, 0.0, 0.0])
                self.mesh = mesh
                self.get_logger().info(f"✅ Mesh loaded: {filename}")
                self.current_state = "IDLE"
                self.resume_state = "IDLE"
            except Exception as e:
                self.get_logger().error(f"Failed to load mesh: {e}")

    def target_cb(self, msg):
        if self.current_state == "FOLLOWING":
            raw_x = np.clip(msg.z + 0.2, self.workspace['x'][0], self.workspace['x'][1])
            raw_y = np.clip(-msg.x, self.workspace['y'][0], self.workspace['y'][1])
            raw_z = np.clip(-msg.y + 0.8, self.workspace['z'][0], self.workspace['z'][1])
            self.target_pose = np.array([raw_x, raw_y, raw_z])
            self.target_z_dir = np.array([0.0, 0.0, -1.0])
            self.last_valid_time = time.time()

    def guided_cb(self, msg):
        if self.current_state == "GUIDED_FOLLOWING" and self.mesh is not None:
            rx = self.surface_bounds['x'][0] + msg.y * (self.surface_bounds['x'][1] - self.surface_bounds['x'][0])
            ry = self.surface_bounds['y'][0] + (1.0 - msg.x) * (self.surface_bounds['y'][1] - self.surface_bounds['y'][0])
            
            origins = np.array([[rx, ry, 2.0]])
            dirs = np.array([[0.0, 0.0, -1.0]])
            locs, _, index_tri = self.mesh.ray.intersects_location(origins, dirs)
            
            if len(locs) > 0:
                surface_z = locs[0][2]
                normal = self.mesh.face_normals[index_tri[0]]
            else:
                closest_pts, _, tri_ids = self.mesh.nearest.on_surface(np.array([[rx, ry, 0.0]]))
                surface_z = closest_pts[0][2]
                normal = self.mesh.face_normals[tri_ids[0]]
                
            self.target_pose = np.array([rx, ry, surface_z]) + normal * self.tcp_z_offset
            self.target_z_dir = -normal
            self.last_valid_time = time.time()

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        parts = msg.data.split(',')
        l, r = parts[0].split(':')[1], parts[1].split(':')[1]
        
        self.gesture_history.append((l, r))
        if len(self.gesture_history) > 10: self.gesture_history.pop(0)
        
        if len(self.gesture_history) == 10:
            count = { "paper_rock": 0, "paper_pointing": 0, "rock_rock": 0, "paper_paper": 0, "rock_scissors": 0 }
            for h in self.gesture_history:
                if h == ("paper", "rock"): count["paper_rock"] += 1
                elif h == ("paper", "pointing"): count["paper_pointing"] += 1
                elif h == ("rock", "rock"): count["rock_rock"] += 1
                elif h == ("paper", "paper"): count["paper_paper"] += 1
                elif h == ("rock", "scissors"): count["rock_scissors"] += 1
            
            if count["rock_rock"] >= 7: 
                if self.current_state not in ["ESTOP", "RESETTING", "NOT READY"]:
                    self.resume_state = self.current_state
                self.current_state = "ESTOP"
                
            elif count["paper_paper"] >= 7 and self.current_state not in ["IDLE", "RESETTING"]: 
                self.current_state = "RESETTING" 
                self.resume_state = "IDLE"
            
            elif count["rock_scissors"] >= 7 and self.current_state in ["IDLE", "ESTOP"]:
                self.current_state = "DEBUG_VERTICAL"
                self.resume_state = "DEBUG_VERTICAL"
                self.needs_ik_reset = True # 💡 모드 진입 시 목표점 리셋 플래그 활성화
                
            elif count["paper_rock"] >= 7: 
                if self.current_state == "IDLE":
                    self.current_state = "FOLLOWING"
                    self.resume_state = "FOLLOWING"
                    self.needs_ik_reset = True
                elif self.current_state == "ESTOP" and self.resume_state == "FOLLOWING":
                    self.last_valid_time = time.time() 
                    self.current_state = "FOLLOWING"
                    self.needs_ik_reset = True
                    
            elif count["paper_pointing"] >= 7: 
                if self.current_state == "IDLE":
                    self.current_state = "GUIDED_FOLLOWING"
                    self.resume_state = "GUIDED_FOLLOWING"
                    self.needs_ik_reset = True
                elif self.current_state == "ESTOP" and self.resume_state == "GUIDED_FOLLOWING":
                    self.last_valid_time = time.time() 
                    self.current_state = "GUIDED_FOLLOWING"
                    self.needs_ik_reset = True

    # -------------------------------------------------------------------------
    # 기구학 및 IK 알고리즘 (레퍼런스와 100% 동일)
    # -------------------------------------------------------------------------
    def unwrap_angle(self, angle):
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle

    def get_fk_6dof(self, q):
        T = np.eye(4)
        for i in range(6):
            ct, st = math.cos(q[i]), math.sin(q[i])
            ca, sa = math.cos(self.alp[i]), math.sin(self.alp[i])
            Ti = np.array([[ct, -st*ca, st*sa, self.a[i]*ct],[st, ct*ca, -ct*sa, self.a[i]*st],[0, sa, ca, self.d[i]],[0, 0, 0, 1]])
            T = T @ Ti
        tcp_pos = T[:3, 3] + T[:3, 2] * self.tool_len
        return tcp_pos, T[:3, :3]

    def calc_geometric_jacobian(self, q):
        J = np.zeros((6, 6))
        T = np.eye(4); trans = [T.copy()]
        for i in range(6):
            ct, st = math.cos(q[i]), math.sin(q[i])
            ca, sa = math.cos(self.alp[i]), math.sin(self.alp[i])
            Ti = np.array([[ct, -st*ca, st*sa, self.a[i]*ct],[st, ct*ca, -ct*sa, self.a[i]*st],[0, sa, ca, self.d[i]],[0, 0, 0, 1]])
            T = T @ Ti
            trans.append(T.copy())
        tcp_pos = trans[-1][:3, 3] + trans[-1][:3, 2] * self.tool_len
        for i in range(6):
            z_i, o_i = trans[i][:3, 2], trans[i][:3, 3]
            J[:3, i] = np.cross(z_i, tcp_pos - o_i) 
            J[3:, i] = z_i 
        return J

    def solve_ik_safe(self, q_current, target_pos, target_z_dir, hint_base=None):
        q = q_current.copy()
        if hint_base is not None:
            diff = self.unwrap_angle(hint_base - q[0])
            q[0] = q[0] + diff * 0.1 
            
        for _ in range(10):
            tcp_pos, R = self.get_fk_6dof(q)
            err_pos = target_pos - tcp_pos
            err_rot = np.cross(R[:, 2], target_z_dir) 
            error = np.concatenate((err_pos, err_rot))
            
            if np.linalg.norm(error) < 0.001: break 
            
            J = self.calc_geometric_jacobian(q)
            J_JT = J @ J.T + 0.05**2 * np.eye(6)
            q += (J.T @ np.linalg.inv(J_JT) @ error * 0.5) 
            
        return np.clip(q, self.q_min, self.q_max) 

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()
        
        # 💡 [핵심 해결 2] 모드 전환 시, 목표점을 로봇의 "현재 위치"로 초기화 (순간이동 방지)
        if self.needs_ik_reset:
            curr_pos, curr_rot = self.get_fk_6dof(self.current_joints)
            self.current_ik_target_pos = curr_pos.copy()
            self.current_ik_target_z_dir = curr_rot[:, 2].copy()
            self.target_pose = curr_pos.copy()
            self.target_z_dir = curr_rot[:, 2].copy()
            self.needs_ik_reset = False
            self.get_logger().info("✅ IK Target Synchronized to Current Position.")

        if self.current_state == "DEBUG_VERTICAL":
            if self.mesh is not None:
                rx, ry = 0.5, 0.0  
                closest_pts, _, tri_ids = self.mesh.nearest.on_surface(np.array([[rx, ry, 0.0]]))
                surface_z = closest_pts[0][2]
                normal = self.mesh.face_normals[tri_ids[0]]
                self.target_pose = np.array([rx, ry, surface_z]) + normal * self.tcp_z_offset
                self.target_z_dir = -normal

        if self.current_state in ["FOLLOWING", "GUIDED_FOLLOWING", "DEBUG_VERTICAL"]:
            if self.current_state in ["FOLLOWING", "GUIDED_FOLLOWING"] and time.time() - self.last_valid_time > 1.5:
                self.current_state = "ESTOP"
            else:
                # 💡 [핵심 해결 3] 가상의 당근을 조금씩 이동 (최대 속도: 1틱당 1cm)
                pos_diff = self.target_pose - self.current_ik_target_pos
                pos_dist = np.linalg.norm(pos_diff)
                max_step = 0.01 # 1틱당 1cm 속도 제한
                if pos_dist > max_step:
                    pos_diff = pos_diff / pos_dist * max_step
                self.current_ik_target_pos += pos_diff
                
                # 방향도 천천히 회전
                z_diff = self.target_z_dir - self.current_ik_target_z_dir
                self.current_ik_target_z_dir += z_diff * 0.05
                self.current_ik_target_z_dir /= np.linalg.norm(self.current_ik_target_z_dir)
                
                hint = math.atan2(self.current_ik_target_pos[1], self.current_ik_target_pos[0])
                self.target_q = self.solve_ik_safe(self.current_joints, self.current_ik_target_pos, self.current_ik_target_z_dir, hint)
                
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralControllerV9()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
