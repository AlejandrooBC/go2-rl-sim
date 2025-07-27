import numpy as np

# Obtain environment and reward cfgs
def get_cfgs():
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

    reward_cfg = {
        "linear_vel_weight": 1.2, # previously 1.0
        "ang_vel_weight": 0.3,
        "height_weight": 0.5,
        "pose_weight": 0.3, # previously 0.25
        "action_rate_weight": 0.3,
        "vertical_vel_weight": 0.25,
        "orientation_weight": 0.3,
        "torque_weight": 0.0005,
        "tracking_sigma": 0.25,
        "alive_bonus": 0.1,
        "duration_bonus": 0.00025 # Not used right now
    }

    return env_cfg, reward_cfg