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
        self.prev_action = np.zeros(self.model.nu)

        # Define homing joint positions (standing pose) with shape = (12,)
        self.homing_pose = np.array([
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8,
            0.0, 0.9, -1.8
        ])

        self.action_scale = 1.0 # Can tune this | how large actions output by policy are (physical joint movements)
        self.target_vel = 0.5 # Track desired velocity

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
        obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel, self.prev_action])
        return obs.astype(np.float32)

    # The core loop of the environment - each time our PPO agent sends an action, this happens:
    def step(self, action):
        action = np.clip(action, -1.0, 1.0) # Clamp action to safe range
        target_pos = self.homing_pose + action * self.action_scale # Action transformation (based on homing pose)

        # Feeding control signal (input) to the robot - array of size 12 (number of actuators on the Go2)
        # The Go2 has 4 legs, 3 actuators per leg (abduction, hip, knee) = 12 total actuators
        self.data.ctrl[:] = target_pos

        # Step the physics forward several times for smoother simulation
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # Construct the updated observation and extract roll, pitch, and yaw
        obs = self._construct_observation()
        rpy = obs[0:3]

        # Extract physical measurements from sim
        forward_vel = self.data.qvel[0]
        vertical_vel = self.data.qvel[2]
        ang_vel = self.data.qvel[3:6]
        forward_pos = self.data.qpos[0]
        z_height = self.data.qpos[2]
        pose_diff = self.data.qpos[7:] - self.homing_pose

        lin_vel_reward = 1.0 - np.abs(forward_vel - self.target_vel)
        pose_penalty = np.sum(np.square(pose_diff)) # Penalizes deviation from homing pose
        height_penalty = np.square(z_height - 0.27) # Penalizes crouching or hopping
        roll_pitch_penalty = rpy[0]**2 + rpy[1]**2 # Penalizes tilt
        vert_vel_penalty = vertical_vel**2 # Penalizes vertical bobbing
        ang_vel_penalty = np.sum(np.square(ang_vel)) # Penalizes rapid rotations
        action_rate_penalty = np.sum(np.square(action - self.prev_action)) # Smooth actuator control

        # Reward function
        reward = (
            3.0 * lin_vel_reward
            - 0.5 * pose_penalty
            - 0.5 * height_penalty
            - 0.3 * roll_pitch_penalty
            - 0.2 * vert_vel_penalty
            - 0.1 * ang_vel_penalty
            - 0.005 * action_rate_penalty
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

        # Small alive bonus if Go2 is within the height range
        if 0.15 < z_height < 0.40:
            reward += 0.1

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
        self.prev_action = np.zeros(self.model.nu) # Reset previous action
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