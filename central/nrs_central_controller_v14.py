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

class CentralControllerV16(Node):
    def __init__(self):
        super().__init__('nrs_central_controller_v16')
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.current_state = "NOT READY"
        self.resume_state = "IDLE" 
        
        self.home_joints = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])
        self.current_joints = self.home_joints.copy()
        self.target_q = self.home_joints.copy()
        
        self.target_pose = np.array([0.5, 0.0, 0.5])
        self.target_z_dir = np.array([0.0, 0.0, -1.0]) 
        
        self.current_ik_target_pos = np.array([0.5, 0.0, 0.5])
        self.current_ik_target_z_dir = np.array([0.0, 0.0, -1.0])
        self.needs_ik_reset = False
        
        self.d = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
        self.a = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])
        self.alp = np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0])
        
        self.tool_len = 0.27 
        self.tcp_z_offset = 0.0 # 툴이 표면에 닿도록 0 유지
        
        self.q_min = np.array([-6.28]*6)
        self.q_max = np.array([ 6.28]*6)
        
        self.workspace = {'x': [0.35, 0.9], 'y': [-0.5, 0.5], 'z': [0.05, 0.6]}
        self.surface_bounds = {'x': [0.4, 0.7], 'y': [-0.3, 0.3]} 
        
        self.mesh = None
        self.gesture_history = [] 
        
        self.cached_surface_z = 0.0
        self.cached_normal = np.array([0.0, 0.0, 1.0])
        
        self.debug_z = 0.3

        self.create_subscription(String, '/hand_gesture', self.gesture_cb, 10)
        self.create_subscription(Point, '/target_pose', self.target_cb, 10)
        self.create_subscription(Point, '/guided_target', self.guided_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(String, '/surface_command', self.surface_cmd_cb, 10)
        
        self.cmd_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.status_pub = self.create_publisher(String, '/robot_status', 10)

        # 💡 [개선 1] 삭제되었던 상태 퍼블리셔(카메라와 통신) 완벽 복구!
        self.timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.last_valid_time = time.time()

        self.get_logger().info("🌟 Central Controller V16 (Vision Sync & Forward Direction Restored) Ready.")

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
            
            new_target = np.array([raw_x, raw_y, raw_z])
            self.target_pose = 0.6 * new_target + 0.4 * self.target_pose
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
                self.cached_surface_z = locs[0][2]
                self.cached_normal = self.mesh.face_normals[index_tri[0]]
                
            safe_target = np.array([rx, ry, self.cached_surface_z]) + self.cached_normal * self.tcp_z_offset
            
            self.target_pose = 0.2 * safe_target + 0.8 * self.target_pose
            self.target_z_dir = -self.cached_normal 
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
                if self.current_state != "DEBUG_VERTICAL":
                    self.current_state = "DEBUG_VERTICAL"
                    self.debug_z = 0.3 
                    self.needs_ik_reset = True
                    
                    if self.mesh is not None:
                        origins = np.array([[0.5, 0.0, 2.0]])
                        dirs = np.array([[0.0, 0.0, -1.0]])
                        locs, _, _ = self.mesh.ray.intersects_location(origins, dirs)
                        if len(locs) > 0: self.cached_surface_z = locs[0][2]
                        else: self.cached_surface_z = 0.0
                        
                    self.get_logger().info(f"🚀 디버그: 앞쪽(파란상자)을 향해 수직 하강합니다!")
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

    def solve_ik_safe(self, q_seed, target_pos, target_z_dir):
        """ 💡 [개선 2] 사라졌던 '180도 좌표 반전' 로직 완벽 복구 (앞으로 무조건 갑니다!) """
        q = q_seed.copy()
        
        # World 프레임 -> IK 프레임으로 180도 반전시켜 IK엔진에 주입
        ik_target_pos = np.array([-target_pos[0], -target_pos[1], target_pos[2]])
        ik_target_z_dir = np.array([-target_z_dir[0], -target_z_dir[1], target_z_dir[2]])
        
        hint_base = math.atan2(ik_target_pos[1], ik_target_pos[0])
        
        diff = self.unwrap_angle(hint_base - q[0])
        q[0] += diff * 0.5 
            
        for _ in range(15): 
            tcp_pos, R = self.get_fk_6dof(q)
            err_pos = ik_target_pos - tcp_pos
            err_rot = np.cross(R[:, 2], ik_target_z_dir) 
            
            error = np.concatenate((err_pos, err_rot))
            if np.linalg.norm(error) < 0.001: break 
            
            J = self.calc_geometric_jacobian(q)
            J_JT = J @ J.T + 0.01 * np.eye(6)
            q += (J.T @ np.linalg.inv(J_JT) @ error * 0.2) 
            
        return np.clip(q, self.q_min, self.q_max) 

    def control_loop(self):
        if self.current_state == "NOT READY": return
        cmd = Float64MultiArray()

        if self.needs_ik_reset:
            # IK 내부가 반전되어 있으므로, 밖에서 쓸 때도 반전된 현재 값을 기준으로 동기화
            curr_pos, curr_rot = self.get_fk_6dof(self.current_joints)
            world_curr_pos = np.array([-curr_pos[0], -curr_pos[1], curr_pos[2]])
            world_curr_z_dir = np.array([-curr_rot[:, 2][0], -curr_rot[:, 2][1], curr_rot[:, 2][2]])
            
            self.current_ik_target_pos = world_curr_pos.copy()
            self.current_ik_target_z_dir = world_curr_z_dir.copy()
            self.target_q = self.current_joints.copy()
            self.needs_ik_reset = False

        if self.current_state == "DEBUG_VERTICAL":
            target_z = self.cached_surface_z + self.tcp_z_offset
            if self.debug_z > target_z:
                self.debug_z -= 0.001
            
            # 파란 상자(X=0.5) 정중앙을 타겟으로 잡음
            target_pos = np.array([0.5, 0.0, self.debug_z])
            target_z_dir = np.array([0.0, 0.0, -1.0])
            
            self.target_q = self.solve_ik_safe(self.target_q, target_pos, target_z_dir)

        elif self.current_state in ["FOLLOWING", "GUIDED_FOLLOWING"]:
            if time.time() - self.last_valid_time > 1.5:
                self.current_state = "ESTOP"
            else:
                pos_diff = self.target_pose - self.current_ik_target_pos
                pos_dist = np.linalg.norm(pos_diff)
                max_step = 0.01 
                if pos_dist > max_step:
                    pos_diff = pos_diff / pos_dist * max_step
                self.current_ik_target_pos += pos_diff
                
                self.current_ik_target_z_dir = self.target_z_dir.copy()
                
                self.target_q = self.solve_ik_safe(self.target_q, self.current_ik_target_pos, self.current_ik_target_z_dir)
                
        elif self.current_state == "RESETTING":
            diff = self.home_joints - self.current_joints
            if np.linalg.norm(diff) < 0.05: self.current_state = "IDLE"
            else: self.target_q = self.current_joints + diff * 0.05
            
        cmd.data = self.target_q.tolist()
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init(); node = CentralControllerV16()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == '__main__': main()
