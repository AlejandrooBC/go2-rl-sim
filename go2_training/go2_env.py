import gymnasium as gym
import numpy as np
import mujoco
from mujoco import MjModel, MjData, viewer
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

# Custom Gymnasium environment for simulating the Unitree Go2 robot using MuJoCo (inherits from Env class)
class UnitreeGo2Env(gym.Env):
    def __init__(self, model_path="../mujoco_menagerie/unitree_go2/scene.xml", render_mode="human"):
        # Load the MuJoCo model and allocate simulation data
        self.model = MjModel.from_xml_path(model_path) # Blueprint of the Go2 model
        self.data = MjData(self.model) # Live state of the Go2 and world - snapshot

        # Simulate action delay
        # self.last_action = np.zeros(self.model.nu)
        # self.simulate_action_latency = False

        # Control frequency: number of MuJoCo physics steps to take per environment step --> for each call to step()
        self.sim_steps = 5 # 20 is 50 Hz control frequency to match the real Go2 (1000/20)

        # Render mode and viewer
        self.render_mode = render_mode
        self.viewer = None

        # Initial Go2 configuration: base pose and joint angles
        self.init_qpos = np.array([
            0, 0, 0.27,  # Base position (x = 0, y = 0, z = 0.27 m) --> slightly above the floor
            1, 0, 0, 0,  # Base orientation quaternion (w, x, y, z) --> robot is upright with no rotation

            # Joint angles for each leg: abduction (side swing), hip (forward/back), knee (extension/flexion)
            0, 0.9, -1.8,  # Front-left
            0, 0.9, -1.8,  # Front-right
            0, 0.9, -1.8,  # Rear-left
            0, 0.9, -1.8  # Rear-right
        ])

        # Initial Go2 velocities: 6 base DOFs + 12 joints = 18 total velocity DOFs --> all start at zero (static spawn)
        self.init_qvel = np.zeros_like(self.data.qvel)

        # Observation and action spaces
        self._obs_template = self._construct_observation()
        obs_dim = len(self._obs_template) # Get dimension of observation vector for Gym's observation space

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

        # Final observation vector: orientation, base motion, joint states
        obs = np.concatenate([rpy, lin_vel, ang_vel, joint_pos, joint_vel])
        return obs.astype(np.float32)

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

        # Apply last action for latency simulation
        # exec_action = self.last_action if self.simulate_action_latency else scaled_action

        # Feeding rescaled control signal (input) to the robot - array of size 12 (number of actuators on the Go2)
        # The Go2 has 4 legs, 3 actuators per leg (abduction, hip, knee) = 12 total actuators
        self.data.ctrl[:] = scaled_action
        # self.last_action = scaled_action.copy()

        # Step the physics forward several times for smoother simulation
        for i in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # Construct the updated observation and extract roll, pitch, and yaw
        obs = self._construct_observation()
        rpy = obs[:3]

        # Define forward velocity, position, and heights
        forward_velocity = self.data.qvel[0]
        forward_position = self.data.qpos[0]
        z_height = self.data.qpos[2]
        target_height = 0.27

        # Reward shaping
        posture_penalty = 0.2 * (rpy[0] ** 2 + rpy[1] ** 2) # Penalize tilt/encourage staying upright (roll, pitch)
        height_penalty = 1.2 * (z_height - target_height) ** 2 # Encourage maintaining target height
        torque_effort = np.sum(np.square(self.data.ctrl)) # Penalize excessive actuator effort
        alive_bonus = 0.2 # Small constant reward to encourage survival

        # Reward function
        reward = 1.3 * forward_velocity - height_penalty - posture_penalty - 0.001 * torque_effort + alive_bonus

        # Episode ends if robot falls
        terminated = bool(z_height < 0.15 or z_height > 0.40)
        truncated = bool(False)

        # Info for Tensorboard logging
        info = {
            "x_position": forward_position,
            "z_height": z_height,
            "x_velocity": forward_velocity,
            "reward": reward
        }

        return obs, reward, terminated, truncated, info

    # Reset sim to a known starting state (called at the beginning of each episode during training/evaluation)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Calls parent class's reset() method

        # Set initial robot pose for the Go2 based on go2.xml
        self.data.qpos[:] = self.init_qpos

        # Set all joint and base velocities to zero - robot not moving at spawn
        self.data.qvel[:] = self.init_qvel

        # Tell MuJoCo to re-calculate everything (kinematics, contacts, etc.) after you manually change the state
        mujoco.mj_forward(self.model, self.data)

        # Reset last action on episode reset
        # self.last_action = np.zeros(self.model.nu)

        return self._construct_observation(), {}

    # Renders the MuJoCo environment for model visualization
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()