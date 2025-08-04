import rclpy
from rclpy.node import Node
import numpy as np
from unitree_go.msg import LowState, LowCmd
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
import os
import time

# Node for sim-to-real deployment of a PPO-trained forward walking policy
class Sim2RealNode(Node):
    def __init__(self):
        # Initialize the ROS2 node with this name
        super().__init__("sim2real_node")

        # Automatically determine the path to this script's directory
        package_path = os.path.dirname(__file__)

        # Model path relative to this package
        model_path = os.path.join(package_path, "walking_policy", "ppo_go2_forward_walk.zip")

        # Load the PPO policy (trained model exported from simulation)
        self.model = PPO.load(model_path)

        # Observation vector (will be filled on each LowState message)
        self.obs = None

        # Initial pose initialization conditions
        self.initializing = True
        self.init_start_time = time.time()

        # Initial stand pose (match sim)
        self.stand_pose = np.array([0.0, 0.9, -1.8,
                                    0.0, 0.9, -1.8,
                                    0.0, 0.9, -1.8,
                                    0.0, 0.9, -1.8])

        # Actuator control ranges used during training (MuJoCo sim)
        self.joint_mins = np.array([-23.7, -23.7, -45.43,
                                    -23.7, -23.7, -45.43,
                                    -23.7, -23.7, -45.43,
                                    -23.7, -23.7, -45.43])
        self.joint_maxs = np.array([23.7, 23.7, 45.43,
                                    23.7, 23.7, 45.43,
                                    23.7, 23.7, 45.43,
                                    23.7, 23.7, 45.43])

        # Actuator control scaling used during training (MuJoCo sim)
        self.mid = 0.5 * (self.joint_mins + self.joint_maxs)
        self.amp = 0.5 * (self.joint_maxs - self.joint_mins)

        # Subscriber to the Go2's /lowstate topic (get observations - match with sim)
        self.create_subscription(LowState, "/lowstate", self.lowstate_callback, 10)

        # Publisher to the Go2's /lowcmd topic (low-level motor commands)
        self.cmd_pub = self.create_publisher(LowCmd, "/lowcmd", 10)

        # Run control callback at 50 Hz (0.02 seconds)
        self.timer = self.create_timer(0.02, self.control_callback)

    # Function to construct the real observation vector (match with sim)
    def lowstate_callback(self, msg: LowState):
        # Convert IMU orientation quaternion [w, x, y, z] -> roll, pitch, yaw
        quat = [
            msg.imu_state.quaternion[0], # w
            msg.imu_state.quaternion[1], # x
            msg.imu_state.quaternion[2], # y
            msg.imu_state.quaternion[3] # z
        ]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)

        # Base linear velocity: currently not available in /lowstate (only acceleration) - placeholder zeros
        lin_vel = np.zeros(3, dtype=np.float32)

        # Base angular velocity from IMU gyroscope (x, y, z)
        ang_vel = np.array(msg.imu_state.gyroscope, dtype=np.float32)

        # Joint positions & velocities (first 12 joints)
        joint_pos = [m.q for m in msg.motor_state[:12]]
        joint_vel = [m.dq for m in msg.motor_state[:12]]

        # Previous action (zeros for now)
        prev_action = np.zeros(12, dtype=np.float32)

        # Combine everything into one observation vector
        self.obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel, prev_action]).astype(np.float32)

    # Function to run the trained policy and continuously send the resulting joint position commands to the Go2
    def control_callback(self):
        # If no observation is available (obs built from /lowstate callback) do nothing
        if self.obs is None:
            return

        # Build /lowcmd message
        cmd_msg = LowCmd()
        cmd_msg.level_flag = 0xFF # Enable low-level motor control

        # Send commands to set the robot to the initial standing pose (1 sec in pose)
        if self.initializing and time.time() - self.init_start_time < 1.0:
            # Send fixed standing pose
            for i in range(12):
                cmd_msg.motor_cmd[i].mode = 0x01
                cmd_msg.motor_cmd[i].q = self.stand_pose[i]
                cmd_msg.motor_cmd[i].dq = 0.0
                cmd_msg.motor_cmd[i].tau = 0.0
                cmd_msg.motor_cmd[i].kp = 20.0
                cmd_msg.motor_cmd[i].kd = 0.5
        # Robot executes the learned policy (after the initial pose)
        else:
            self.initializing = False

            # Use the loaded PPO policy to predict an action (12 joint target positions for the Go2)
            action, _ = self.model.predict(self.obs, deterministic=True)

            # Action scaling based on actuator joint limits from MuJoCo training
            scaled_action = self.mid + action * self.amp

            # For each of the 12 actuated joints
            for i in range(12):
                cmd_msg.motor_cmd[i].mode = 0x01 # Position control mode
                cmd_msg.motor_cmd[i].q = scaled_action[i] # Desired joint angle from the policy
                cmd_msg.motor_cmd[i].dq = 0.0 # No velocity control
                cmd_msg.motor_cmd[i].tau = 0.0 # No torque control
                cmd_msg.motor_cmd[i].kp = 20.0 # Proportional gain - how strongly motor tries to move target angle
                cmd_msg.motor_cmd[i].kd = 0.5 # Derivative gain - how strongly it resists velocity changes (damping)

        # Publish the built command to /lowcmd
        self.cmd_pub.publish(cmd_msg)

# Set up and execution
def main(args=None):
    rclpy.init(args=args) # Initialize ROS2 Python library (rclpy)
    node = Sim2RealNode() # Create an instance of the node class
    rclpy.spin(node) # Start the ROS2 event loop
    node.destroy_node() # Clean up resources for node when shutting down
    rclpy.shutdown() # Shut down ROS2 for this program

# Entry point
if __name__ == "__main__":
    main()