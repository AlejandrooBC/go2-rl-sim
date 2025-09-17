import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData, viewer
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from training_cfgs_dom_rand import get_cfgs
from collections import deque
from rewards_dom_rand import (
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
                 render_mode="human", dr_cfg=None, env_cfg=None, reward_cfg=None):
        # Load the MuJoCo model and allocate simulation data
        self.model = MjModel.from_xml_path(model_path) # Blueprint of the Go2 model
        self.data = MjData(self.model) # Live state of the Go2 and world - snapshot

        # Load the domain randomization, environment, and reward cfgs
        if dr_cfg is None or env_cfg is None or reward_cfg is None:
            self.dr_cfg, self.env_cfg, self.reward_cfg = get_cfgs()
        else:
            self.dr_cfg = dr_cfg
            self.env_cfg = env_cfg
            self.reward_cfg = reward_cfg

        # Domain randomization state
        self.dr_on = bool(self.dr_cfg.get("enabled", False)) # Determine whether domain randomization is on or off
        self.rng = np.random.default_rng()
        self.dt = float(self.model.opt.timestep)

        # IDs we need (body and floor)
        self._base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Store nominal values once (to reapply scales)
        self._geom_friction_nominal = self.model.geom_friction[:, 0].copy()  # Slide component
        self._body_mass_nominal = self.model.body_mass.copy()

        # Placeholders for sampled DR parameters
        self._friction_scale = 1.0
        self._mass_scale = 1.0
        self._motor_scale = 1.0
        self._latency_steps = 0

        # Action latency FIFO (length is set on reset after sampling)
        self._act_fifo = deque([np.zeros(self.model.nu, dtype=np.float32)], maxlen=1)

        # IMU (gyro) perturbations — your obs uses only ang_vel (3D)
        self._imu_bias_gyro = np.zeros(3, dtype=np.float32)
        self._imu_noise_gyro_std = np.zeros(3, dtype=np.float32)

        # Push state (external force in world frame)
        self._push_world_force = np.zeros(3, dtype=np.float32)
        self._push_steps_left = 0

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

        # DR: corrupt gyro
        if self.dr_on:
            ang_vel = ang_vel + self._imu_bias_gyro + self.rng.normal(0.0, self._imu_noise_gyro_std, size=3)

        # Joint positions and velocities for all 12 actuators
        joint_pos = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]

        # Previous action
        prev_action = self.prev_action

        # Final observation vector: orientation, base motion, joint states
        obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel, prev_action])
        return obs.astype(np.float32)

    # Sample domain randomization parameters per episode
    def _sample_domain_randomization(self):
        if not self.dr_on:
            return
        U = lambda a: self.rng.uniform(a[0], a[1])

        # Physics scales
        self._fric_scale  = U(self.dr_cfg.get("friction", [1.0, 1.0]))
        self._mass_scale  = U(self.dr_cfg.get("mass_scale", [1.0, 1.0]))
        self._motor_scale = U(self.dr_cfg.get("motor_strength", [1.0, 1.0]))

        # Latency steps (integer)
        lat = self.dr_cfg.get("latency_steps", [0, 0])
        self._latency_steps = int(self.rng.integers(lat[0], lat[1] + 1))
        self._act_fifo = deque(
            [np.zeros(self.model.nu, dtype=np.float32) for _ in range(self._latency_steps + 1)],
            maxlen=self._latency_steps + 1
        )

        # IMU gyro bias/noise (3D; match your obs)
        b_lo, b_hi = self.dr_cfg.get("imu_bias_gyro", [0.0, 0.0])
        self._imu_bias_gyro = self.rng.uniform(b_lo, b_hi, size=3).astype(np.float32)

        n_lo, n_hi = self.dr_cfg.get("imu_noise_gyro_std", [0.0, 0.0])
        std = float(self.rng.uniform(n_lo, n_hi))
        self._imu_noise_gyro_std[:] = std

        # Random push decision per episode
        self._push_world_force[:] = 0.0
        self._push_steps_left = 0
        if self.rng.random() < float(self.dr_cfg.get("push_prob", 0.0)):
            Fmin, Fmax = self.dr_cfg.get("push_force_N", [0.0, 0.0])
            Tmin, Tmax = self.dr_cfg.get("push_duration", [0.0, 0.0])
            mag = float(self.rng.uniform(Fmin, Fmax))
            dur = float(self.rng.uniform(Tmin, Tmax))
            steps = max(1, int(round(dur / self.dt)))
            theta = float(self.rng.uniform(0.0, 2.0 * np.pi))
            self._push_world_force[:] = [mag * np.cos(theta), mag * np.sin(theta), 0.0]
            self._push_steps_left = steps

    # Applies domain randomization to the simulation
    def _apply_domain_randomization(self):
        if not self.dr_on:
            # Restore defaults if DR is off
            self.model.geom_friction[:, 0] = self._geom_friction_nominal
            self.model.body_mass[:]       = self._body_mass_nominal
            mujoco.mj_forward(self.model, self.data)
            return

        # Randomize floor friction (do not double-count feet)
        if self._floor_geom_id != -1:
            self.model.geom_friction[self._floor_geom_id, 0] = \
                self._geom_friction_nominal[self._floor_geom_id] * self._fric_scale
        else:
            # Fallback: scale all geoms slide friction
            self.model.geom_friction[:, 0] = self._geom_friction_nominal * self._fric_scale

        # Global mass scale
        self.model.body_mass[:] = self._body_mass_nominal * self._mass_scale

        # Recompute
        mujoco.mj_forward(self.model, self.data)

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
        # Map normalized action [-1,1] → actuator ctrlrange, include motor strength DR
        ctrl_range = self.model.actuator_ctrlrange
        mid = 0.5 * (ctrl_range[:, 0] + ctrl_range[:, 1])
        amp = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        scaled_now = mid + (action * self._motor_scale) * amp

        # Latency: send the oldest command in the FIFO
        self._act_fifo.append(scaled_now)
        applied_ctrl = self._act_fifo[0]
        self.data.ctrl[:] = applied_ctrl

        # Step the physics forward several times for smoother simulation
        for i in range(self.sim_steps):
            # DR: external horizontal push on base body (world frame)
            if self._push_steps_left > 0 and self._base_body_id != -1:
                self.data.xfrc_applied[self._base_body_id, 0:3] = self._push_world_force
                self._push_steps_left -= 1
            else:
                self.data.xfrc_applied[self._base_body_id, 0:3] = 0.0

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
                action_rate_penalty(self, applied_ctrl, self.reward_cfg) +
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
        self.prev_action = applied_ctrl.copy()

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
            "r_action_rate_penalty": action_rate_penalty(self, applied_ctrl, self.reward_cfg),
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

        # DR: sample + apply at episode start
        if self.dr_on and self.dr_cfg.get("rerandomize_on_reset", True):
            self._sample_domain_randomization()
            self._apply_domain_randomization()

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