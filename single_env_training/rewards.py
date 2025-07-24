import numpy as np

# Reward for maintaining base x-velocity near target velocity
def linear_velocity_tracking(env, reward_cfg):
    target_vel = 0.5 # Forward walking target velocity of 0.5 m/s
    actual_vel = env.data.qvel[0] # X-direction base linear velocity
    vel_error = (target_vel - actual_vel) ** 2 # Measures the difference between the actual velocity and the target
    reward = np.exp(-vel_error / reward_cfg["tracking_sigma"]) * reward_cfg["linear_vel_weight"]
    return reward

# Reward for maintaining low angular velocity in x, y, z
def angular_velocity_tracking(env, reward_cfg):
    ang_vel = env.data.qvel[3:6] # Angular velocity in rad/s
    error = np.linalg.norm(ang_vel)
    reward = np.exp(-error / reward_cfg["tracking_sigma"]) * reward_cfg["ang_vel_weight"]
    return reward

# Reward for maintaining base z-height near target height
def height_penalty(env, reward_cfg):
    actual_z = env.data.qpos[2] # Base height
    error = (env.target_height - actual_z) ** 2 # Measures the difference between the actual height and the target
    reward = np.exp(-error / reward_cfg["tracking_sigma"]) * reward_cfg["height_weight"]
    return reward

# Reward for maintaining joints similar to the default initialization pose
def pose_similarity(env, reward_cfg):
    error = np.linalg.norm(env.data.qpos[7:] - reward_cfg["default_joint_pose"]) # Skip base pose
    reward = np.exp(-error / reward_cfg["tracking_sigma"]) * reward_cfg["pose_weight"]
    return reward

# Penalize rapid changes in action
def action_rate_penalty(env, action, reward_cfg):
    if hasattr(env, "prev_action"):
        diff = action - env.prev_action
    else:
        diff = np.zeros_like(action)
    penalty = np.linalg.norm(diff)
    return np.exp(-penalty / reward_cfg["tracking_sigma"]) * reward_cfg["action_rate_weight"]

# Penalize vertical (z-axis) base velocity
def vertical_velocity_penalty(env, reward_cfg):
    vertical_vel = env.data.qvel[2] # Z-direction base linear velocity
    penalty = vertical_vel ** 2
    return -penalty * reward_cfg["vertical_vel_weight"]

# Penalize roll and pitch deviations from upright
def orientation_penalty(env, reward_cfg):
    # Convert quaternion to Euler angles
    quat = env.data.qpos[3:7] # Assuming quaternion is at this position
    w, x, y, z = quat
    # Roll (x), Pitch (y), Yaw (z)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    penalty = roll**2 + pitch**2
    return -penalty * reward_cfg["orientation_weight"]