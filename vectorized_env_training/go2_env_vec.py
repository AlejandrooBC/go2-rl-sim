import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData, viewer
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

# Custom Gymnasium environment for simulating the Unitree Go2 robot using MuJoCo (inherits from Env class)
class UnitreeGo2Env(gym.Env):
    def __init__(self, model_path="../mujoco_menagerie/unitree_go2/scene.xml", render_mode=None):
        # Load the MuJoCo model and allocate simulation data
        self.model = MjModel.from_xml_path(model_path) # Blueprint of the Go2 model
        self.data = MjData(self.model) # Live state of the Go2 and world - snapshot
        self.render_mode = render_mode
        self.viewer = None if render_mode != "human" else viewer.launch_passive(self.model, self.data)

        # Number of MuJoCo physics steps to take per environment step --> for each call to step()
        self.sim_steps = 5
        self.step_counter = 0
        self.prev_x = 0.0
        self.prev_vel = 0.0
        self.target_height = 0.27

        # Initial Go2 configuration: base pose and joint angles
        self.init_qpos = np.array([
            0, 0, 0.27, # Base position (x = 0, y = 0, z = 0.27 m) --> slightly above the floor
            1, 0, 0, 0, # Base orientation quaternion (w, x, y, z) --> robot is upright with no rotation

            # Joint angles for each leg: abduction (side swing), hip (forward/back), knee (extension/flexion)
            0, 0.9, -1.8, # Front-left
            0, 0.9, -1.8, # Front-right
            0, 0.9, -1.8, # Rear-left
            0, 0.9, -1.8 # Rear-right
        ])

        # Initial Go2 velocities: 6 base DOFs + 12 joints = 18 total velocity DOFs --> all start at zero (static spawn)
        self.init_qvel = np.zeros_like(self.data.qvel)

        # Get dimension of observation vector for Gym's observation space
        obs_dim = len(self._construct_observation())

        # Define the observation space: what the robot/agent sees
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define the action space: what the robot/agent can do --> one control input per actuator
        # If the shape = (12,) then the policy will output 12 numbers between (low, high) per step
        # Box = continuous values with (low, high) bounds | nu = number of actuators
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)

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

        # Final observation vector: orientation, base motion, joint states, previous action
        obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel])
        return obs.astype(np.float32)

    # The core loop of the environment - each time our PPO agent sends an action, this happens:
    def step(self, action):
        action = np.clip(action, -1.0, 1.0) # Clamp action to safe range

        # Ctrl_range = Array of shape (n_actuators, 2) = (12, 2)
        # Defines min and max control signal (torque, position, target, etc.) for each actuator
        ctrl_range = self.model.actuator_ctrlrange

        # Mid is the midpoint of each actuator's range, amp is the half range
        mid = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1])  # Mid = 0.5 * (min values + max values)
        amp = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])  # Amp = 0.5 * (max values - min values)
        scaled_action = mid + action * amp  # Rescales action ∈ [-1, 1] to scaled_action ∈ [min, max]

        # Feeding control signal (input) to the robot - array of size 12 (number of actuators on the Go2)
        # The Go2 has 4 legs, 3 actuators per leg (abduction, hip, knee) = 12 total actuators
        self.data.ctrl[:] = scaled_action

        # Step the physics forward several times for smoother simulation
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # Construct the updated observation and extract roll, pitch, and yaw
        obs = self._construct_observation()
        rpy = obs[0:3]

        # Extract physical measurements from sim
        forward_vel = self.data.qvel[0]
        vertical_vel = self.data.qvel[2]
        forward_pos = self.data.qpos[0]
        z_height = self.data.qpos[2]

        # Rewards and penalties
        posture_penalty = 0.5 * (rpy[0] ** 2 + rpy[1] ** 2) # Penalize tilt/encourage staying upright (roll, pitch)
        height_penalty = 0.5 * (z_height - self.target_height) ** 2 # Encourage maintaining target height
        torque_effort = np.sum(np.square(self.data.ctrl)) # Penalize excessive actuator effort
        alive_bonus = 1.0 # Small constant reward to encourage survival
        duration_reward = 0.8 * self.step_counter if forward_vel > 0.1 else 0.0 # Small reward to keep moving

        # Penalize sudden forward acceleration
        forward_acc = forward_vel - self.prev_vel
        acc_penalty = 0.8 * (forward_acc ** 2)
        self.prev_vel = forward_vel

        # Reward function
        reward = (
            14.0 * forward_vel
            - acc_penalty
            - height_penalty
            - posture_penalty
            - 0.001 * torque_effort
            + alive_bonus
            + duration_reward
        )

        # NaN/Inf safeguard (debug print if unstable)
        if not np.isfinite(reward):
            print(f"[NaN Warning] reward={reward}, z={z_height}, vel={forward_vel}, obs[0:3]={rpy}")

        # Compute delta_x for logging only
        delta_x = forward_pos - self.prev_x
        self.prev_x = forward_pos
        self.step_counter += 1

        # Episode ends if robot falls
        terminated = bool(z_height < 0.15 or z_height > 0.40)
        truncated = bool(False)

        # Apply fall penalty if the Go2 falls
        if terminated:
            reward -= 1

        # Info for Tensorboard logging
        info = {
            "x_position": forward_pos,
            "z_height": z_height,
            "x_velocity": forward_vel,
            "z_velocity": vertical_vel,
            "delta_x": delta_x,
            "steps_alive": self.step_counter,
            "reward": reward
        }

        return obs, reward, terminated, truncated, info

    # Reset sim to a known starting state (called at the beginning of each episode during training/evaluation)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self.init_qpos # Set initial robot pose for the Go2 based on go2.xml
        self.data.qvel[:] = self.init_qvel # Set all joint and base velocities to zero - robot not moving at spawn
        self.prev_x = 0.0 # Reset delta_x tracker
        self.prev_vel = 0.0 # Reset previous velocity
        self.step_counter = 0 # Reset step counter
        mujoco.mj_forward(self.model, self.data) # Re-calculate states
        return self._construct_observation(), {}

    # Renders the MuJoCo visualization
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()