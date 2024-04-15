import argparse

import tempfile

import gymnasium as gym
import torch
from cleanrl.dqn_atari import QNetwork
from huggingface_hub import HfApi, hf_hub_download
from torch import jit, nn
import numpy as np
API = HfApi()


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-entity", type=str, default="cleanrl",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--hf-repository", type=str, default="",
        help="the huggingface repo (e.g., cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1)")
    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    args = parser.parse_args()
    # fmt: on
    return args


class QNetwork(nn.Module):
    def __init__(self, num_actions, epsilon):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )
        self.num_actions = num_actions
        self.epsilon = epsilon

    def forward(self, observation):
        q_values = self.network(observation / 255.0)
        actions = torch.argmax(q_values, dim=1)
        random_action = torch.randint(0, self.num_actions, (observation.shape[0],))
        return torch.where(torch.rand_like(actions.float()) < self.epsilon, random_action, actions)


if __name__ == "__main__":
    args = parse_args()
    if not args.hf_repository:
        args.hf_repository = f"{args.hf_entity}/{args.env_id}-dqn_atari-seed{args.seed}"
    model_path = hf_hub_download(repo_id=args.hf_repository, filename=f"dqn_atari.cleanrl_model")
    env = gym.make(args.env_id)
    agent = QNetwork(env.action_space.n, epsilon=0.05)
    agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    dummy_input = torch.from_numpy(np.random.rand(1, 4, 84, 84).astype(np.float32))
    traced_agent = jit.trace(agent.eval(), dummy_input, check_trace=False)
    frozen_agent = jit.freeze(traced_agent)
    frozen_agent = jit.optimize_for_inference(frozen_agent)

    with tempfile.NamedTemporaryFile() as tmp:
        jit.save(frozen_agent, tmp)

        API.upload_file(
            path_or_fileobj=tmp.name,
            path_in_repo="agent.pt",
            repo_id=args.hf_repository,
            repo_type="model",
            commit_message="Add `agent.pt` and tag the environment",
            commit_description="This commit adds a model that is compatible with the 🥇 [Open RL Leaderboard](https://huggingface.co/spaces/open-rl-leaderboard/leaderboard) 🥇.",
            create_pr=True,
        )

# python cleanrl_push_dqn_atari_agent.py --hf-entity qgallouedec --env-id QbertNoFrameskip-v4
