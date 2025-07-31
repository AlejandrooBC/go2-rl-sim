import rclpy
from rclpy.node import Node
import numpy as np
from unitree_go.msg import LowState, LowCmd
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO

class Sim2RealNode(Node):
    def __init__(self):
        super().__init__("sim2real_node")

        # Load PPO policy
        model_path = '/path/to/your/trained_models_single/ppo_go2_11M_best_steps.zip'
        self.model = PPO.load(model_path)

        # Observation buffer
        self.obs = None

        # Subscriber to /lowstate
        self.create_subscription(LowState, '/lowstate', self.lowstate_callback, 10)

        # Publisher to /lowcmd
        self.cmd_pub = self.create_publisher(LowCmd, '/lowcmd', 10)

        # Control loop timer (50 Hz)
        self.timer = self.create_timer(0.02, self.control_callback)

    def lowstate_callback(self, msg: LowState):
        # Orientation quaternion [w, x, y, z]
        quat = [
            msg.imu_state.quaternion[0],
            msg.imu_state.quaternion[1],
            msg.imu_state.quaternion[2],
            msg.imu_state.quaternion[3]
        ]
        rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)

        # Base angular velocity
        ang_vel = np.array(msg.imu_state.gyroscope, dtype=np.float32)

        # Approx base linear velocity (use accel if needed)
        lin_vel = np.zeros(3, dtype=np.float32)

        # Joint positions & velocities (first 12 joints)
        joint_pos = [m.q for m in msg.motor_state[:12]]
        joint_vel = [m.dq for m in msg.motor_state[:12]]

        # Previous action (zeros for now)
        prev_action = np.zeros(12, dtype=np.float32)

        self.obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel, prev_action]).astype(np.float32)

    def control_callback(self):
        if self.obs is None:
            return

        # Run policy
        action, _ = self.model.predict(self.obs, deterministic=True)

        # Build /lowcmd
        cmd_msg = LowCmd()
        cmd_msg.level_flag = 0xFF  # enable motor control
        for i in range(12):
            cmd_msg.motor_cmd[i].mode = 0x01  # position control mode
            cmd_msg.motor_cmd[i].q = action[i]
            cmd_msg.motor_cmd[i].dq = 0.0
            cmd_msg.motor_cmd[i].tau = 0.0
            cmd_msg.motor_cmd[i].kp = 20.0
            cmd_msg.motor_cmd[i].kd = 0.5

        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Sim2RealNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()