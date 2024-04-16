import json
import os
import tempfile
import argparse

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.repocard import RepoCard
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.huggingface.huggingface_utils import load_from_hf
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sf_examples.atari.atari_utils import atari_env_by_name
from sf_examples.atari.train_atari import register_atari_components
from torch import nn

API = HfApi()

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", nargs="?", default=None)


class Agent(nn.Module):
    def __init__(self, actor_critic, deterministic):
        super().__init__()
        self.actor_critic = actor_critic
        self.deterministic = deterministic

    def forward(self, obs):
        normalized_obs = self.actor_critic.normalize_obs({"obs": obs})
        policy_outputs = self.actor_critic(normalized_obs, torch.tensor([[0.0]]))
        if self.deterministic:
            action_distribution = self.actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)
        else:
            actions = policy_outputs["actions"]
        return actions


def make_and_upload_agent(repo_id):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load the model from the Hugging Face Hub
        load_from_hf(temp_dir, repo_id)
        user_id, model_id = repo_id.split("/")

        # Open temp_dir/model_id/config.json, or temp_dir/model_id/cfg.json
        config_file = (
            f"{temp_dir}/{model_id}/config.json"
            if os.path.exists(f"{temp_dir}/{model_id}/config.json")
            else f"{temp_dir}/{model_id}/cfg.json"
        )
        with open(config_file) as f:
            cfg = AttrDict(json.load(f))

        cfg.train_dir = temp_dir
        cfg.user_id = user_id
        cfg.experiment = model_id
        cfg.policy_index = 0
        cfg.eval_deterministic = False

        register_atari_components()
        env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))

        # reset call ruins the demo recording for VizDoom
        if hasattr(env.unwrapped, "reset_on_init"):
            env.unwrapped.reset_on_init = False

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        name_prefix = {"latest": "checkpoint", "best": "best"}[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, cfg.policy_index), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, "cpu")
        actor_critic.load_state_dict(checkpoint_dict["model"])
        agent = Agent(actor_critic, cfg.eval_deterministic)

        # Trace and optimize the module
        dummy_input = torch.tensor(np.array([env.observation_space["obs"].sample()]))
        traced_agent = torch.jit.trace(agent.eval(), dummy_input)
        frozen_agent = torch.jit.freeze(traced_agent)

        # Save and upload
        commits = []
        torch.jit.save(frozen_agent, f"{temp_dir}/agent.pt")
        commit = CommitOperationAdd(path_in_repo="agent.pt", path_or_fileobj=f"{temp_dir}/agent.pt")
        commits.append(commit)

        card = RepoCard.load(f"{cfg.user_id}/{cfg.experiment}")
        env_id = atari_env_by_name(cfg.env).env_id
        if env_id not in card.data.tags:
            card.data.tags.append(env_id)
            card.save(f"{temp_dir}/README.md")
            commit = CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=f"{temp_dir}/README.md")
            commits.append(commit)

        API.create_commit(
            repo_id=f"{cfg.user_id}/{cfg.experiment}",
            operations=commits,
            commit_message="Add `agent.pt` and tag the environment",
            commit_description="This commit adds a model that is compatible with the [🥇 Open RL Leaderboard 🥇](https://huggingface.co/spaces/open-rl-leaderboard/leaderboard).",
            repo_type="model",
            # create_pr=True,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    make_and_upload_agent(args.repo_id)
