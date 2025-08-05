import torch, zipfile, torch.nn as nn

zip_path = "/home/unitree/sim2real_ws/src/go2_policy_deploy/go2_policy_deploy/walking_policy/ppo_go2_forward_walk.zip"
export_path = "/home/unitree/sim2real_ws/src/go2_policy_deploy/go2_policy_deploy/walking_policy/ppo_go2_forward_walk_torchscript.pt"

# --- Load raw policy weights ---
with zipfile.ZipFile(zip_path, 'r') as archive:
    with archive.open('policy.pth', 'r') as f:
        state_dict = torch.load(f, map_location='cpu')

# Extract dimensions from loaded weights
obs_dim = state_dict['mlp_extractor.policy_net.0.weight'].shape[1]
hidden = state_dict['mlp_extractor.policy_net.0.weight'].shape[0]
hidden2 = state_dict['mlp_extractor.policy_net.2.weight'].shape[0]
act_dim = state_dict['action_net.weight'].shape[0]

# --- Minimal policy network ---
class DummyPolicy(nn.Module):
    def __init__(self, obs_dim, hidden, hidden2, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden2),
            nn.Tanh(),
            nn.Linear(hidden2, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # from SB3
    def forward(self, x):
        mean = self.net(x)
        return mean, self.log_std.exp().expand_as(mean)

policy = DummyPolicy(obs_dim, hidden, hidden2, act_dim)

# Map SB3 keys -> our dummy network
mapping = {
    'mlp_extractor.policy_net.0.weight': 'net.0.weight',
    'mlp_extractor.policy_net.0.bias': 'net.0.bias',
    'mlp_extractor.policy_net.2.weight': 'net.2.weight',
    'mlp_extractor.policy_net.2.bias': 'net.2.bias',
    'action_net.weight': 'net.4.weight',
    'action_net.bias': 'net.4.bias',
    'log_std': 'log_std'
}
new_state_dict = {}
for k, v in state_dict.items():
    if k in mapping:
        new_state_dict[mapping[k]] = v
policy.load_state_dict(new_state_dict, strict=False)

# --- TorchScript tracing ---
example_obs = torch.zeros((1, obs_dim))
traced = torch.jit.trace(policy, example_obs)
torch.jit.save(traced, export_path)
print(f"TorchScript model saved at {export_path}")