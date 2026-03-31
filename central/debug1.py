import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point, WrenchStamped
from sensor_msgs.msg import JointState
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import time
import trimesh
import os

class UltimateDebugControllerV9(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_debug')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_state = "NOT READY"
        self.resume_state = "IDLE" 
        
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.target_z_dir = np.array([0.0, 0.0, -1.0]) 
        
        self.d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
        self.a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])
        self.alp = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
        
        # 💡 사진 분석으로 찾았던 14.5cm 유지
        self.tool_len = 0.145 
        self.tcp_z_offset = 0.0 
        
        self.q_min = np.array([-6.28]*6)
        self.q_max = np.array([ 6.28]*6)
        
        self.workspace = {'x': [0.35, 0.9], 'y': [-0.5, 0.5], 'z': [0.05, 0.6]}
        self.surface_bounds = {'x': [0.4, 0.7], 'y': [-0.3, 0.3]} 
        
        self.mesh = None
        self.gesture_history = [] 
        
        self.debug_z = 0.3 
        self.cached_surface_z = 0.0 
        
        # 💡 [핵심 추가] 힘 센서 데이터 변수
        self.current_force_z = 0.0
        self.contact_detected = False

        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(Point, '/guided_target', self.guided_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        # 💡 [핵심 추가] 브릿지가 보내는 물리적 힘(Force) 데이터를 구독합니다.
        self.create_subscription(WrenchStamped, '/ft_sensor', self.ft_cb, qos_profile_sensor_data)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("🌟 Ultimate Debug V9 (Auto Force Brake Calibration) Ready.")

    def ft_cb(self, msg):
        self.current_force_z = msg.wrench.force.z

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            mesh_path = os.path.join(self.script_dir, "visual", filename)
            try:
                mesh = trimesh.load(mesh_path)
                mesh.apply_translation([0.5, 0.0, 0.0])
                self.mesh = mesh
                self.current_state = "IDLE"
                self.get_logger().info(f"✅ Mesh loaded: {filename}")
            except Exception as e:
                self.get_logger().error(f"Failed to load mesh: {e}")

    def target_cb(self, msg):
        if self.current_state == "FOLLOWING":
            self.target_pose = np.array([np.clip(msg.z + 0.2, self.workspace['x'][0], self.workspace['x'][1]),
                                         np.clip(-msg.x, self.workspace['y'][0], self.workspace['y'][1]),
                                         np.clip(-msg.y + 0.8, self.workspace['z'][0], self.workspace['z'][1])])
            self.target_z_dir = np.array([0.0, 0.0, -1.0])

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
                self.current_state = "ESTOP"
            elif count["paper_paper"] >= 7: 
                self.current_state = "RESETTING" 
            elif count["rock_scissors"] >= 7 and self.current_state in ["IDLE", "ESTOP"]:
                if self.current_state != "DEBUG_VERTICAL":
                    self.current_state = "DEBUG_VERTICAL"
                    self.debug_z = 0.3 
                    self.contact_detected = False # 💡 재진입 시 충돌 감지 초기화
                    self.get_logger().info("🚀 [오토 캘리브레이션 시작] 로봇이 하강하다가 표면을 감지하면 스스로 멈춥니다!")
            elif count["paper_rock"] >= 7 and self.current_state in ["IDLE", "ESTOP"]: 
                self.current_state = "FOLLOWING"
            elif count["paper_pointing"] >= 7 and self.current_state in ["IDLE", "ESTOP"]: 
                self.current_state = "GUIDED_FOLLOWING"

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

    def solve_ik_ultra_safe(self, q_seed, target_pos, target_z_dir):
        q = q_seed.copy()
        ik_target_pos = np.array([-target_pos[0], -target_pos[1], target_pos[2]])
        ik_target_z_dir = np.array([-target_z_dir[0], -target_z_dir[1], target_z_dir[2]])
        
        hint_base = math.atan2(ik_target_pos[1], ik_target_pos[0])
        
        diff = self.unwrap_angle(hint_base - q[0])
        q[0] += diff * 0.5 
            
        for _ in range(30):
            tcp_pos, R = self.get_fk_6dof(q)
            err_pos = ik_target_pos - tcp_pos
            err_rot = np.cross(R[:, 2], ik_target_z_dir) 
            
            error = np.concatenate((err_pos, err_rot))
            if np.linalg.norm(error) < 0.001: break 
            
            J = self.calc_geometric_jacobian(q)
            J_JT = J @ J.T + 0.01 * np.eye(6)
            q += (J.T @ np.linalg.inv(J_JT) @ error * 0.2)
            
        return q

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()
        
        if self.current_state == "DEBUG_VERTICAL":
            
            # Trimesh 표면 높이 탐색 (수학적 기준)
            if self.mesh is not None:
                origins = np.array([[0.5, 0.0, 2.0]])
                dirs = np.array([[0.0, 0.0, -1.0]])
                locs, _, _ = self.mesh.ray.intersects_location(origins, dirs)
                if len(locs) > 0: self.cached_surface_z = locs[0][2]
            
            # 💡 [핵심] 오토 브레이크 및 영점 측정 로직
            if not self.contact_detected:
                self.debug_z -= 0.001 # 1틱당 1mm 하강
                
                # 만약 센서가 충격(2.0N 이상)을 감지했다면!
                if abs(self.current_force_z) > 2.0:
                    self.contact_detected = True
                    
                    # 현재 멈춰선 로봇의 관절로 '진짜 물리적 Z 위치'를 알아냅니다.
                    real_tcp, _ = self.get_fk_6dof(self.current_joints)
                    physical_z = real_tcp[2] 
                    
                    perfect_offset = physical_z - self.cached_surface_z
                    self.debug_z = physical_z # 더 이상 뚫지 않고 현재 높이에서 멈춤
                    
                    self.get_logger().info(f"💥 [표면 안착!] 힘 센서가 {self.current_force_z:.2f}N을 감지하여 브레이크를 밟았습니다.")
                    self.get_logger().info(f"👉 V18 메인 컨트롤러에 입력할 완벽한 tcp_z_offset 은 [{perfect_offset:.4f}] 입니다!")
            
            target_pos = np.array([0.5, 0.0, self.debug_z])
            target_z_dir = np.array([0.0, 0.0, -1.0])
            
            self.target_q = self.solve_ik_ultra_safe(self.target_q, target_pos, target_z_dir)
            
            if not self.contact_detected and int(time.time()*10) % 5 == 0:
                self.get_logger().info(f"⬇️ 로봇이 하강하며 표면(힘)을 찾는 중... [현재 Z: {self.debug_z:.4f} m | 힘: {self.current_force_z:.2f}N]")

        elif self.current_state in ["FOLLOWING", "GUIDED_FOLLOWING"]:
            self.target_q = self.solve_ik_ultra_safe(self.target_q, self.target_pose, self.target_z_dir)
                
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = UltimateDebugControllerV9()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
