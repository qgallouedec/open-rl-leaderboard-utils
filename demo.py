import argparse

import gymnasium as gym
import torch
from huggingface_hub import HfApi, metadata_save
from torch import nn, optim
from torch.distributions import Categorical

argparser = argparse.ArgumentParser()
argparser.add_argument("--hf_user", type=str, required=True)
args = argparser.parse_args()

# Setup the 🤗 API
api = HfApi()

# Environment setup
env_id = "CartPole-v1"
env = gym.make(env_id, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)

# Policy setup
policy = nn.Sequential(
    nn.Linear(4, 128),
    nn.Dropout(p=0.6),
    nn.ReLU(),
    nn.Linear(128, 2),
    nn.Softmax(-1),
)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Training loop
global_step = 0
for episode_idx in range(100):
    log_probs = torch.zeros((env.spec.max_episode_steps + 1))
    returns = torch.zeros((env.spec.max_episode_steps + 1))
    observation, info = env.reset()
    terminated = truncated = False
    step = 0
    while not terminated and not truncated:
        probs = policy(torch.tensor(observation))
        distribution = Categorical(probs)  # Create distribution
        action = distribution.sample()  # Sample action
        log_probs[step] = distribution.log_prob(action)  # Store log probability
        action = action.cpu().numpy()  # Convert to numpy array
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
        global_step += 1
        returns[:step] += 0.99 ** torch.flip(torch.arange(step), (0,)) * reward  # return = sum(gamma^i * reward_i)

    episodic_return = info["episode"]["r"][0]
    print(f"Episode: {episode_idx}  Global step: {global_step}  Episodic return: {episodic_return:.2f}")

    batch_returns = returns[:step]
    batch_log_probs = log_probs[:step]
    batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 10**-5)
    policy_loss = torch.sum(-batch_log_probs * batch_returns)
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()


# Setup the agent
class Agent(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observations):
        probs = self.policy(observations)
        distribution = Categorical(probs)
        return distribution.sample()


agent = Agent(policy)  # instantiate the agent

# A few tests to check if the agent is working
observations = torch.randn(env.observation_space.shape).unsqueeze(0)  # dummy batch of observations
actions = agent(observations)
actions = actions.numpy()[0]
assert env.action_space.contains(actions)

# Save model along with its card
metadata_save("model_card.md", {"tags": ["reinforcement-learning", env_id]})
dummy_input = torch.tensor(env.observation_space.sample()).unsqueeze(0)  # dummy batch of observations
agent = torch.jit.trace(agent.eval(), dummy_input)
agent = torch.jit.freeze(agent)  # required for for the model not to depend on the training library
torch.jit.save(agent, "agent.pt")

# # Upload model and card to the 🤗 Hub
repo_id = f"{args.hf_user}/REINFORCE-CartPole-v1"  # can be any name
model_path = api.create_repo(repo_id, repo_type="model")
api.upload_file(path_or_fileobj="agent.pt", path_in_repo="agent.pt", repo_id=repo_id)
api.upload_file(path_or_fileobj="model_card.md", path_in_repo="README.md", repo_id=repo_id)
