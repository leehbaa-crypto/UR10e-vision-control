import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import numpy as np
import time
import trimesh
import os

class CentralControllerV4(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v4')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.current_state = "NOT READY"
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.target_rot = np.eye(3) 
        self.filtered_target = self.target_pose.copy()
        self.last_sent_target = self.target_pose.copy()
        
        self.alpha = 0.8        
        self.deadzone = 0.002   
        self.z_offset = 0.03    
        
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

        self.timer = self.create_timer(0.033, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.last_valid_time = time.time()

        self.get_logger().info("🌟 Central Controller V4 (Resume/Restart Support) Ready.")

    def publish_status(self):
        msg = String(); msg.data = self.current_state
        self.status_pub.publish(msg)

    def surface_cmd_cb(self, msg):
        if msg.data.startswith("LAUNCH:"):
            filename = msg.data.split(":")[1]
            mesh_path = os.path.join(self.script_dir, "visual", filename)
            try:
                self.mesh = trimesh.load(mesh_path)
                self.get_logger().info(f"✅ Mesh loaded for Guide Mode: {filename} from {mesh_path}")
                self.current_state = "IDLE"
            except Exception as e:
                self.get_logger().error(f"❌ Failed to load mesh: {e} | Path: {mesh_path}")

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
                
                safe_target = np.array([rx, ry, surface_z]) + normal * self.z_offset
                self.target_pose = self.alpha * safe_target + (1.0 - self.alpha) * self.target_pose
                
                Z_tool = -normal
                Y_tool = np.array([0, 1, 0])
                X_tool = np.cross(Y_tool, Z_tool)
                X_tool /= np.linalg.norm(X_tool)
                Y_tool = np.cross(Z_tool, X_tool)
                
                self.target_rot = np.column_stack((X_tool, Y_tool, Z_tool))

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
            
            # [기능 1 반영] ESTOP 상태에서도 다른 상태로 복귀할 수 있도록 조건 완화 (Resume/Restart)
            if count["rock_rock"] >= 7: 
                self.current_state = "ESTOP"
            elif count["paper_rock"] >= 7 and self.current_state in ["IDLE", "ESTOP"]: 
                if self.current_state == "ESTOP": self.last_valid_time = time.time() # Resume시 즉각 종료 방지
                self.current_state = "FOLLOWING"
            elif count["paper_pointing"] >= 7 and self.current_state in ["IDLE", "FOLLOWING", "ESTOP"]: 
                if self.current_state == "ESTOP": self.last_valid_time = time.time() # Resume시 즉각 종료 방지
                self.current_state = "GUIDED_FOLLOWING"
            elif count["paper_paper"] >= 7 and self.current_state not in ["IDLE", "RESETTING"]: 
                self.current_state = "RESETTING" # 홈으로 복귀 (Restart)

    def fast_geometric_ik(self, target, current_q):
        q = current_q.copy()
        for _ in range(8):
            T = np.eye(4); z_axes = [np.array([0, 0, 1])]; p_joints = [np.zeros(3)]
            for i in range(6):
                a, d, alpha = self.dh[i]['a'], self.dh[i]['d'], self.dh[i]['alpha']
                ct, st, ca, sa = np.cos(q[i]), np.sin(q[i]), np.cos(alpha), np.sin(alpha)
                Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
                T = T @ Ti
                z_axes.append(T[:3, 2]); p_joints.append(T[:3, 3])
            
            p_ee = T[:3, 3]; err = target - p_ee
            if np.linalg.norm(err) < 0.001: break
            
            J = np.zeros((3, 6))
            for i in range(6): J[:, i] = np.cross(z_axes[i], (p_ee - p_joints[i]))
            
            step = J.T @ np.linalg.inv(J @ J.T + 0.001 * np.eye(3)) @ err
            q += np.clip(step, -0.2, 0.2)
            q = np.clip(q, self.q_min, self.q_max) 
        return q

    def get_fk_6dof(self, q):
        T = np.eye(4)
        for i in range(6):
            a, d, alpha = self.dh[i]['a'], self.dh[i]['d'], self.dh[i]['alpha']
            ct, st = np.cos(q[i]), np.sin(q[i]); ca, sa = np.cos(alpha), np.sin(alpha)
            Ti = np.array([[ct, -st*ca, st*sa, a*ct], [st, ct*ca, -ct*sa, a*st], [0, sa, ca, d], [0, 0, 0, 1]])
            T = T @ Ti
        return T[:3, 3], T[:3, :3]

    def numerical_ik_6dof_dls(self, target_p, target_R, current_q):
        q = current_q.copy()
        lambda_sq = 0.05 ** 2
        
        for _ in range(10):
            p, R = self.get_fk_6dof(q)
            err_p = target_p - p
            err_o = 0.5 * (np.cross(R[:,0], target_R[:,0]) + np.cross(R[:,1], target_R[:,1]) + np.cross(R[:,2], target_R[:,2]))
            
            err = np.hstack((err_p, err_o))
            if np.linalg.norm(err) < 1e-3: break
            
            J = np.zeros((6, 6))
            delta = 1e-5
            for i in range(6):
                qp = q.copy(); qp[i] += delta
                pp, Rp = self.get_fk_6dof(qp)
                J[:3, i] = (pp - p) / delta
                err_o_p = 0.5 * (np.cross(R[:,0], Rp[:,0]) + np.cross(R[:,1], Rp[:,1]) + np.cross(R[:,2], Rp[:,2]))
                J[3:, i] = err_o_p / delta
                
            JJt = J @ J.T
            J_dls = J.T @ np.linalg.inv(JJt + lambda_sq * np.eye(6))
            step = J_dls @ err
            q += np.clip(step, -0.1, 0.1)
            
        return np.clip(q, self.q_min, self.q_max) 

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()
        
        if self.current_state == "FOLLOWING":
            if time.time() - self.last_valid_time > 1.5:
                self.current_state = "ESTOP"
            else:
                self.target_q = self.fast_geometric_ik(self.target_pose, self.current_joints)
                
        elif self.current_state == "GUIDED_FOLLOWING":
            self.target_q = self.numerical_ik_6dof_dls(self.target_pose, self.target_rot, self.current_joints)
            
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralControllerV4()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
