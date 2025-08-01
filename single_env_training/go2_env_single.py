import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData, viewer
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from training_cfgs import get_cfgs
from rewards import (
    linear_velocity_tracking,
    angular_velocity_tracking,
    height_penalty,
    pose_similarity,
    action_rate_penalty,
    vertical_velocity_penalty,
    orientation_penalty,
    torque_penalty
)

# Custom Gymnasium environment for simulating the Unitree Go2 robot using MuJoCo (inherits from Env class)
class UnitreeGo2Env(gym.Env):
    def __init__(self, model_path="../mujoco_menagerie/unitree_go2/scene.xml",
                 render_mode="human", env_cfg=None, reward_cfg=None):
        # Load the MuJoCo model and allocate simulation data
        self.model = MjModel.from_xml_path(model_path) # Blueprint of the Go2 model
        self.data = MjData(self.model) # Live state of the Go2 and world - snapshot

        # Load the environment and reward cfgs
        if env_cfg is None or reward_cfg is None:
            self.env_cfg, self.reward_cfg = get_cfgs()
        else:
            self.env_cfg = env_cfg
            self.reward_cfg = reward_cfg

        # Initializing trackers, targets, and step counter
        self.prev_x = 0.0 # Track displacement between steps
        self.step_counter = 0 # Track the number of steps per episode
        self.target_height = self.env_cfg["target_height"]

        # Number of MuJoCo physics steps to take per environment step --> for each call to step()
        self.sim_steps = self.env_cfg["sim_steps"]
        self.render_mode = render_mode
        self.viewer = None

        # Initial Go2 configuration: base pose and joint angles
        self.init_qpos = self.env_cfg["initial_pose"]

        # Initial Go2 velocities: 6 base DOFs + 12 joints = 18 total velocity DOFs --> all start at zero (static spawn)
        self.init_qvel = np.zeros_like(self.data.qvel)

        # Initial Go2 previous action
        self.prev_action = np.zeros(self.model.nu)

        # Observation and action spaces
        self._obs_template = self._construct_observation()
        obs_dim = len(self._obs_template)

        # Define the observation space: what the robot/agent sees
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32)

        # Define the action space: what the robot/agent can do --> one control input per actuator
        # If the shape = (12,) then the policy will output 12 numbers between (low, high) per step
        self.action_space = spaces.Box( # Box = continuous values with (low, high) bounds
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,), # nu = number of actuators
            dtype=np.float32)

    # Constructs the observation the robot/agent will receive
    # Returns vector: [roll, pitch, yaw] + [linear vel xyz] + [angular vel xyz] + [joint pos 12] + [joint vel 12]
    def _construct_observation(self):
        # Base orientation quaternion: [w, x, y, z]
        quat = self.data.qpos[3:7]

        # Convert to [roll, pitch, yaw] using scipy (requires [x, y, z, w] order)
        rpy = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)

        lin_vel = self.data.qvel[0:3] # Base linear velocity in x, y, z (world frame)
        ang_vel = self.data.qvel[3:6] # Base angular velocity in x, y, z

        # Joint positions and velocities for all 12 actuators
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        # Previous action
        prev_action = self.prev_action

        # Final observation vector: orientation, base motion, joint states
        obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel, prev_action])
        return obs.astype(np.float32)

    # Function to check for episode termination conditions
    def is_healthy(self):
        # Use existing orientation calculation
        quat = self.data.qpos[3:7]
        roll, pitch, yaw = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_euler("xyz", degrees=False)
        current_height = self.data.qpos[2]

        roll_threshold = 0.35 # Adjust the threshold (rad) - previously 0.5
        pitch_threshold = 0.35 # Adjust the threshold (rad) - previously 0.5
        min_height = self.env_cfg["termination_height_range"][0]
        max_height = self.env_cfg["termination_height_range"][1]

        if (abs(roll) > roll_threshold or abs(pitch) > pitch_threshold or
                current_height < min_height or current_height > max_height):
            return False # Go2 episode terminates
        else:
            return True # Go2 episode continues (remains alive)

    # The core loop of the environment - each time our PPO agent sends an action, this happens:
    def step(self, action):
        action = np.clip(action, -1.0, 1.0) # Clamp action to safe range

        # Ctrl_range = Array of shape (n_actuators, 2) = (12, 2)
        # Defines min and max control signal (torque, position, target, etc.) for each actuator
        ctrl_range = self.model.actuator_ctrlrange

        # Mid is the midpoint of each actuator's range, amp is the half range
        mid = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1]) # Mid = 0.5 * (min values + max values)
        amp = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0]) # Amp = 0.5 * (max values - min values)
        scaled_action = mid + action * amp # Rescales action ∈ [-1, 1] to scaled_action ∈ [min, max]

        # Feeding rescaled control signal (input) to the robot - array of size 12 (number of actuators on the Go2)
        # The Go2 has 4 legs, 3 actuators per leg (abduction, hip, knee) = 12 total actuators
        self.data.ctrl[:] = scaled_action

        # Step the physics forward several times for smoother simulation
        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # Construct the updated observation and extract roll, pitch, and yaw
        obs = self._construct_observation()

        # Define forward velocity, position, and heights
        forward_velocity = self.data.qvel[0]
        forward_position = self.data.qpos[0]
        z_height = self.data.qpos[2]
        lateral_position = self.data.qpos[1]

        # Reward function
        reward = (
                linear_velocity_tracking(self, self.reward_cfg) +
                angular_velocity_tracking(self, self.reward_cfg) +
                height_penalty(self, self.reward_cfg) +
                pose_similarity(self, self.reward_cfg) +
                action_rate_penalty(self, scaled_action, self.reward_cfg) +
                vertical_velocity_penalty(self, self.reward_cfg) +
                orientation_penalty(self, self.reward_cfg) +
                torque_penalty(self, self.reward_cfg) +
                (self.reward_cfg["alive_bonus"] if forward_velocity > 0.2 else 0.0)
        )

        # Compute delta_x for logging only
        delta_x = forward_position - self.prev_x
        self.prev_x = forward_position

        # Increment step counter
        self.step_counter += 1

        # Episode ends if termination conditions are met
        healthy = self.is_healthy()
        terminated = bool(not healthy)
        truncated = bool(False)
        if terminated:
            reward -= 1

        # Store previous action for next observation
        self.prev_action = scaled_action.copy()

        # Info for Tensorboard logging
        info = {
            "x_position": forward_position,
            "z_height": z_height,
            "x_velocity": forward_velocity,
            "delta_x": delta_x,
            "steps_alive": self.step_counter,
            "reward": reward,
            "lateral_position": lateral_position,
            "r_linear_velocity": linear_velocity_tracking(self, self.reward_cfg),
            "r_angular_velocity": angular_velocity_tracking(self, self.reward_cfg),
            "r_height_penalty": height_penalty(self, self.reward_cfg),
            "r_pose_similarity": pose_similarity(self, self.reward_cfg),
            "r_action_rate_penalty": action_rate_penalty(self, scaled_action, self.reward_cfg),
            "r_vertical_velocity_penalty": vertical_velocity_penalty(self, self.reward_cfg),
            "r_orientation_penalty": orientation_penalty(self, self.reward_cfg),
            "r_torque_penalty": torque_penalty(self, self.reward_cfg)
        }

        return obs, reward, terminated, truncated, info

    # Reset sim to a known starting state (called at the beginning of each episode during training/evaluation)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Calls parent class's reset() method

        # Set initial robot pose for the Go2 based on go2.xml
        self.data.qpos[:] = self.init_qpos

        # Set all joint and base velocities to zero - robot not moving at spawn
        self.data.qvel[:] = self.init_qvel

        # Reset delta_x tracker
        self.prev_x = 0.0

        # Reset step counter
        self.step_counter = 0

        # Reset previous action
        self.prev_action = np.zeros(self.model.nu)

        # Tell MuJoCo to re-calculate everything (kinematics, contacts, etc.) after you manually change the state
        mujoco.mj_forward(self.model, self.data)

        return self._construct_observation(), {}

    # Renders the MuJoCo environment for model visualization
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()