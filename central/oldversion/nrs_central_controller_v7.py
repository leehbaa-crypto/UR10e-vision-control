import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time
import trimesh
import os

class CentralControllerV6_1(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v6')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.current_state = "NOT READY"
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.target_z_dir = np.array([0.0, 0.0, -1.0]) 
        
        self.filtered_target = self.target_pose.copy()
        self.last_sent_target = self.target_pose.copy()
        
        self.alpha = 0.8        
        self.deadzone = 0.002   
        self.tcp_z_offset = -0.01 
        
        self.q_min = np.array([-6.28, -3.14, -3.14, -6.28, -6.28, -6.28])
        self.q_max = np.array([ 6.28,  0.00,  3.14,  6.28,  6.28,  6.28])
        self.workspace = {'x': [0.35, 0.9], 'y': [-0.5, 0.5], 'z': [0.05, 0.6]}
        self.surface_bounds = {'x': [0.4, 0.7], 'y': [-0.3, 0.3]} 
        
        self.dh = [
            {'a': 0,        'd': 0.1807,  'alpha': 1.570796},
            {'a': -0.6127,  'd': 0,       'alpha': 0},
            {'a': -0.57155, 'd': 0,       'alpha': 0},
            {'a': 0,        'd': 0.17415, 'alpha': 1.570796},
            {'a': 0,        'd': 0.11985, 'alpha': -1.570796},
            {'a': 0,        'd': 0.11655, 'alpha': 0}
        ]
        
        self.mesh = None
        self.gesture_history = [] 

        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(Point, '/guided_target', self.guided_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.last_valid_time = time.time()

        self.get_logger().info("🌟 Central Controller V6.1 (Geometric Jacobian Fixed) Ready.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            mesh_path = os.path.join(self.script_dir, "visual", filename)
            try:
                self.mesh = trimesh.load(mesh_path)
                self.mesh.apply_translation([0.5, 0.0, 0.0])
                self.current_state = "IDLE"
            except Exception as e:
                self.get_logger().error(f"Failed to load mesh: {e}")

    def target_cb(self, msg):
        if self.current_state == "FOLLOWING":
            raw_x = np.clip(msg.z + 0.2, self.workspace['x'][0], self.workspace['x'][1])
            raw_y = np.clip(-msg.x, self.workspace['y'][0], self.workspace['y'][1])
            raw_z = np.clip(-msg.y + 0.8, self.workspace['z'][0], self.workspace['z'][1])
            
            new_target = np.array([raw_x, raw_y, raw_z])
            if np.linalg.norm(new_target - self.last_sent_target) > self.deadzone:
                self.filtered_target = self.alpha * new_target + (1.0 - self.alpha) * self.filtered_target
                self.target_pose = self.filtered_target
                self.last_sent_target = new_target
                self.last_valid_time = time.time()
                self.target_z_dir = np.array([0.0, 0.0, -1.0])

    def guided_cb(self, msg):
        if self.current_state == "GUIDED_FOLLOWING" and self.mesh is not None:
            rx = self.surface_bounds['x'][0] + msg.y * (self.surface_bounds['x'][1] - self.surface_bounds['x'][0])
            ry = self.surface_bounds['y'][0] + (1.0 - msg.x) * (self.surface_bounds['y'][1] - self.surface_bounds['y'][0])
            
            ray_origins = np.array([[rx, ry, 2.0]])
            ray_dirs = np.array([[0.0, 0.0, -1.0]])
            locations, index_ray, index_tri = self.mesh.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_dirs)
            
            if len(locations) > 0:
                normal = self.mesh.face_normals[index_tri[0]]
                surface_z = locations[0][2]
                
                safe_target = np.array([rx, ry, surface_z]) + normal * self.tcp_z_offset
                self.target_pose = self.alpha * safe_target + (1.0 - self.alpha) * self.target_pose
                self.target_z_dir = -normal

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def gesture_cb(self, msg):
        parts = msg.data.split(',')
        l, r = parts[0].split(':')[1], parts[1].split(':')[1]
        
        self.gesture_history.append((l, r))
        if len(self.gesture_history) > 10: self.gesture_history.pop(0)
        
        if len(self.gesture_history) == 10:
            count = { "paper_rock": 0, "paper_pointing": 0, "rock_rock": 0, "paper_paper": 0 }
            for h in self.gesture_history:
                if h == ("paper", "rock"): count["paper_rock"] += 1
                elif h == ("paper", "pointing"): count["paper_pointing"] += 1
                elif h == ("rock", "rock"): count["rock_rock"] += 1
                elif h == ("paper", "paper"): count["paper_paper"] += 1
            
            if count["rock_rock"] >= 7: 
                self.current_state = "ESTOP"
            elif count["paper_paper"] >= 7 and self.current_state not in ["IDLE", "RESETTING"]: 
                self.current_state = "RESETTING" 
            elif count["paper_rock"] >= 7 and self.current_state in ["IDLE", "ESTOP"]: 
                if self.current_state == "ESTOP": self.last_valid_time = time.time() 
                self.current_state = "FOLLOWING"
            elif count["paper_pointing"] >= 7 and self.current_state in ["IDLE", "ESTOP"]: 
                if self.current_state == "ESTOP": self.last_valid_time = time.time() 
                self.current_state = "GUIDED_FOLLOWING"

    def get_fk_6dof(self, q):
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = self.dh[i]['a'], self.dh[i]['d'], self.dh[i]['alpha']
            ct, st = np.cos(q[i]), np.sin(q[i]); ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
            T = T @ Ti
        return T[:3, 3], T[:3, :3]

    def calc_geometric_jacobian(self, q):
        """ 💡 [레퍼런스 코드 이식] 정확한 기하학적 자코비안 계산 """
        J = np.zeros((6, 6))
        T = np.eye(4)
        trans = [T.copy()]
        for i in range(6):
            a, d, alpha = self.dh[i]['a'], self.dh[i]['d'], self.dh[i]['alpha']
            ct, st = np.cos(q[i]), np.sin(q[i])
            ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
            T = T @ Ti
            trans.append(T.copy())
            
        tcp_pos = trans[-1][:3, 3]
        for i in range(6):
            z_i, o_i = trans[i][:3, 2], trans[i][:3, 3]
            J[:3, i] = np.cross(z_i, tcp_pos - o_i) # 선속도 자코비안
            J[3:, i] = z_i # 각속도 자코비안
        return J

    def geometric_ik_align_z(self, target_p, target_z_dir, current_q):
        """ 💡 [레퍼런스 코드 이식] 기하학적 자코비안을 이용한 안전한 5-DOF 역기구학 """
        q = current_q.copy()
        
        target_angle = np.arctan2(target_p[1], target_p[0])
        diff = target_angle - q[0]
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        q[0] += diff * 0.1 
        
        for _ in range(10):
            p, R = self.get_fk_6dof(q)
            err_p = target_p - p
            err_o = np.cross(R[:, 2], target_z_dir) 
            error = np.concatenate((err_p, err_o))
            
            if np.linalg.norm(error) < 1e-3: break 
            
            # 수학적으로 완벽한 기하학적 자코비안 사용
            J = self.calc_geometric_jacobian(q)
            J_JT = J @ J.T + 0.05**2 * np.eye(6)
            step = J.T @ np.linalg.inv(J_JT) @ error
            q += np.clip(step, -0.1, 0.1)
            
        return np.clip(q, self.q_min, self.q_max) 

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()
        
        if self.current_state in ["FOLLOWING", "GUIDED_FOLLOWING"]:
            if self.current_state == "FOLLOWING" and time.time() - self.last_valid_time > 1.5:
                self.current_state = "ESTOP"
            else:
                self.target_q = self.geometric_ik_align_z(self.target_pose, self.target_z_dir, self.current_joints)
                
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralControllerV6_1()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
