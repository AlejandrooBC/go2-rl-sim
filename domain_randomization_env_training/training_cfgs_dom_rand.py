import numpy as np

# Obtain environment and reward cfgs
def get_cfgs():
    # Domain randomization
    dr_cfg = {
        "enabled": True,
        "rerandomize_on_reset": True,

        # Physics
        "friction": [0.4, 1.2], # Multiplies slide friction
        "mass_scale": [0.90, 1.10], # Global body-mass scale
        "motor_strength": [0.85, 1.15], # Scales action magnitude

        # Latency (in control steps)
        "latency_steps": [0, 2], # 0-2 step delay FIFO

        # IMU (gyro) bias/noise
        "imu_bias_gyro": [-0.05, 0.05], # Rad/s bias per axis
        "imu_noise_gyro_std": [0.0, 0.02], # Rad/s Gaussian noise (per axis)

        # Random pushes (external forces)
        "push_prob": 0.10, # Probability of pushing the robot per episode
        "push_force_N": [0.0, 70.0], # Horizontal force magnitude (N)
        "push_duration": [0.05, 0.20], # Push duration (seconds)
    }

    # Environment configurations
    env_cfg = {
        "sim_steps": 6,
        "termination_height_range": [0.20, 0.40], # Previous min height was 0.15
        "target_height": 0.27,

        "initial_pose": np.array([
            0, 0, 0.27, # Base position (x = 0, y = 0, z = 0.27 m) --> slightly above the floor
            1, 0, 0, 0, # Base orientation quaternion (w, x, y, z) --> robot is upright with no rotation

            # Joint angles for each leg: abduction (side swing), hip (forward/back), knee (extension/flexion)
            0, 0.9, -1.8, # Front-left
            0, 0.9, -1.8, # Front-right
            0, 0.9, -1.8, # Rear-left
            0, 0.9, -1.8 # Rear-right
        ]),

        "default_joint_pose": np.array([
            # Joint angles for each leg: abduction (side swing), hip (forward/back), knee (extension/flexion)
            0, 0.9, -1.8, # Front-left
            0, 0.9, -1.8, # Front-right
            0, 0.9, -1.8, # Rear-left
            0, 0.9, -1.8 # Rear-right
        ])
    }

    # Reward weights and constants
    reward_cfg = {
        "linear_vel_weight": 1.2, # Previously 1.0
        "ang_vel_weight": 0.3,
        "height_weight": 0.5,
        "pose_weight": 0.3, # Previously 0.25
        "action_rate_weight": 0.3,
        "vertical_vel_weight": 0.25,
        "orientation_weight": 0.3,
        "torque_weight": 0.0005,
        "tracking_sigma": 0.25,
        "alive_bonus": 0.1,
        "duration_bonus": 0.00025 # Not used right now
    }

    return dr_cfg, env_cfg, reward_cfg