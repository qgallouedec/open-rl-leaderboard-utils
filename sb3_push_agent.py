import argparse
import importlib
import os
import pickle
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import rl_zoo3.import_envs  # noqa: F401 pylint: disable=unused-import
import torch as th
import yaml
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.repocard import RepoCard
from huggingface_sb3 import EnvironmentName, ModelName, ModelRepoId, load_from_hub
from requests.exceptions import HTTPError
from rl_zoo3 import ALGOS, get_latest_run_id, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.utils import StoreDict, get_model_path
from torch import nn

API = HfApi()

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", nargs="?", default=None)
parser.add_argument("--env", help="environment ID", type=EnvironmentName, default="CartPole-v1")
# parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl-trained-agents")
parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
parser.add_argument("--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)")
parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
parser.add_argument(
    "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
)
parser.add_argument(
    "--load-checkpoint",
    type=int,
    help="Load checkpoint instead of last model if available, " "you must pass the number of timesteps corresponding to it",
)
parser.add_argument(
    "--load-last-checkpoint",
    action="store_true",
    default=False,
    help="Load last checkpoint instead of last model if available",
)
parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
parser.add_argument(
    "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
parser.add_argument(
    "--gym-packages",
    type=str,
    nargs="+",
    default=[],
    help="Additional external Gym environment package modules to import",
)
parser.add_argument(
    "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
)
parser.add_argument("--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues")
parser.add_argument(
    "-P",
    "--progress",
    action="store_true",
    default=False,
    help="if toggled, display a progress bar using tqdm and rich",
)


class Agent(nn.Module):
    def __init__(self, policy, mean_obs, var_obs, epsilon, clip_obs, deterministic):
        super().__init__()
        self.policy = policy
        self.mean_obs = th.tensor(mean_obs) if mean_obs is not None else None
        self.var_obs = th.tensor(var_obs) if var_obs is not None else None
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.deterministic = False

    def forward(self, observations):
        if self.mean_obs is not None:
            observations = (observations - self.mean_obs) / th.sqrt(self.var_obs + self.epsilon)
            observations = th.clip(observations, -self.clip_obs, self.clip_obs)
        return self.policy._predict(observations, deterministic=self.deterministic)


def maybe_upgrade_env(env_id: EnvironmentName) -> str:
    # Most agent have been trained with the v2-v3. Sometimes, deploying an agent trained with v2-v3 in v4 works.
    # So we tag the agent with the v4 version of the environment
    mujoco_ids = [
        "Ant",
        "HalfCheetah",
        "Hopper",
        "Humanoid",
        "HumanoidStandup",
        "InvertedDoublePendulum",
        "InvertedPendulum",
        "Pusher",
        "Reacher",
        "Swimmer",
        "Walker2d",
    ]
    for mujoco_id in mujoco_ids:
        if mujoco_id in env_id:
            return f"{mujoco_id}-v4"
    return env_id


def download_from_hub(
    algo: str,
    env_name: EnvironmentName,
    exp_id: int,
    folder: str,
    organization: str,
    repo_name: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Try to load a model from the Huggingface hub
    and save it following the RL Zoo structure.
    Default repo name is {organization}/{algo}-{env_id}
    where repo_name = {algo}-{env_id}

    :param algo: Algorithm
    :param env_name: Environment name
    :param exp_id: Experiment id
    :param folder: Log folder
    :param organization: Huggingface organization
    :param repo_name: Overwrite default repository name
    :param force: Allow overwritting the folder
        if it already exists.
    """

    model_name = ModelName(algo, env_name)

    if repo_name is None:
        repo_name = model_name  # Note: model name is {algo}-{env_name}

    # Note: repo id is {organization}/{repo_name}
    repo_id = ModelRepoId(organization, repo_name)
    print(f"Downloading from https://huggingface.co/{repo_id}")

    checkpoint = load_from_hub(repo_id, model_name.filename)
    config_path = load_from_hub(repo_id, "config.yml")

    # If VecNormalize, download
    try:
        vec_normalize_stats = load_from_hub(repo_id, "vec_normalize.pkl")
    except HTTPError:
        print("No normalization file")
        vec_normalize_stats = None

    saved_args = load_from_hub(repo_id, "args.yml")
    env_kwargs = load_from_hub(repo_id, "env_kwargs.yml")
    train_eval_metrics = load_from_hub(repo_id, "train_eval_metrics.zip")

    if exp_id == 0:
        exp_id = get_latest_run_id(os.path.join(folder, algo), env_name) + 1
    # Sanity checks
    if exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_name}_{exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    # Check that the folder does not exist
    log_folder = Path(log_path)
    if log_folder.is_dir():
        if force:
            print(f"The folder {log_path} already exists, overwritting")
            # Delete the current one to avoid errors
            shutil.rmtree(log_path)
        else:
            raise ValueError(
                f"The folder {log_path} already exists, use --force to overwrite it, "
                "or choose '--exp-id 0' to create a new folder"
            )

    print(f"Saving to {log_path}")
    # Create folder structure
    os.makedirs(log_path, exist_ok=True)
    config_folder = os.path.join(log_path, env_name)
    os.makedirs(config_folder, exist_ok=True)

    # Copy config files and saved stats
    shutil.copy(checkpoint, os.path.join(log_path, f"{env_name}.zip"))
    shutil.copy(saved_args, os.path.join(config_folder, "args.yml"))
    shutil.copy(config_path, os.path.join(config_folder, "config.yml"))
    shutil.copy(env_kwargs, os.path.join(config_folder, "env_kwargs.yml"))
    if vec_normalize_stats is not None:
        shutil.copy(vec_normalize_stats, os.path.join(config_folder, "vecnormalize.pkl"))

    # Extract monitor file and evaluation file
    with zipfile.ZipFile(train_eval_metrics, "r") as zip_ref:
        zip_ref.extractall(log_path)


def make_and_upload_agent(args):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Going through custom gym packages to let them register in the global registory
        for env_module in args.gym_packages:
            importlib.import_module(env_module)

        env_name: EnvironmentName = args.env
        algo = args.algo
        user_id, model_id = args.repo_id.split("/")

        try:
            _, model_path, log_path = get_model_path(
                args.exp_id,
                temp_dir,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )
        except (AssertionError, ValueError) as e:
            # Special case for rl-trained agents
            # auto-download from the hub
            # if "rl-trained-agents" not in temp_dir:
            #     raise e
            # else:
            print("Pretrained model not found, trying to download it from sb3 Huggingface hub: https://huggingface.co/sb3")
            # Auto-download
            download_from_hub(
                algo=algo,
                env_name=env_name,
                exp_id=args.exp_id,
                folder=temp_dir,
                organization=user_id,
                repo_name=model_id,
                force=False,
            )
            # Try again
            _, model_path, log_path = get_model_path(
                args.exp_id,
                temp_dir,
                algo,
                env_name,
                args.load_best,
                args.load_checkpoint,
                args.load_last_checkpoint,
            )

        print(f"Loading {model_path}")

        # Off-policy algorithm only support one env for now
        off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

        is_atari = ExperimentManager.is_atari(env_name.gym_id)
        is_minigrid = ExperimentManager.is_minigrid(env_name.gym_id)

        stats_path = os.path.join(log_path, env_name)
        hyperparams, maybe_stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

        # load env_kwargs if existing
        env_kwargs = {}
        args_path = os.path.join(log_path, env_name, "args.yml")
        if os.path.isfile(args_path):
            with open(args_path) as f:
                loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
                if loaded_args["env_kwargs"] is not None:
                    env_kwargs = loaded_args["env_kwargs"]
        # overwrite with command line arguments
        if args.env_kwargs is not None:
            env_kwargs.update(args.env_kwargs)

        mean_obs = var_obs = epsilon = clip_obs = None
        if maybe_stats_path is not None:
            if hyperparams["normalize"]:
                print("Loading running average")
                print(f"with params: {hyperparams['normalize_kwargs']}")
                path_ = os.path.join(stats_path, "vecnormalize.pkl")
                if os.path.exists(path_):
                    with open(path_, "rb") as file_handler:
                        vec_norm = pickle.load(file_handler)
                    if vec_norm.norm_obs:
                        mean_obs = vec_norm.obs_rms.mean
                        var_obs = vec_norm.obs_rms.var
                        epsilon = vec_norm.epsilon
                        clip_obs = vec_norm.clip_obs

        kwargs = dict(seed=args.seed)
        if algo in off_policy_algos:
            # Dummy buffer size as we don't need memory to enjoy the trained agent
            kwargs.update(dict(buffer_size=1))

        # Check if we are running python 3.8+
        # we need to patch saved model under python 3.6/3.7 to load them
        newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

        custom_objects = {}
        if newer_python_version or args.custom_objects:
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }

        model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=args.device, **kwargs)

        # Deterministic by default except for atari games
        stochastic = args.stochastic or (is_atari or is_minigrid) and not args.deterministic
        deterministic = not stochastic

        agent = Agent(model.policy, mean_obs, var_obs, epsilon, clip_obs, deterministic)

        dummy_input = th.tensor(np.array([model.observation_space.sample()]))
        traced_agent = th.jit.trace(agent.eval(), dummy_input)
        frozen_agent = th.jit.freeze(traced_agent)

        # Save and upload
        th.jit.save(frozen_agent, f"{temp_dir}/agent.pt")
        commits = []
        commit = CommitOperationAdd(path_in_repo="agent.pt", path_or_fileobj=f"{temp_dir}/agent.pt")
        commits.append(commit)

        card = RepoCard.load(args.repo_id)
        env_id = maybe_upgrade_env(args.env)
        if env_id not in card.data.tags:
            card.data.tags.append(env_id)
            card.save(f"{temp_dir}/README.md")
            commit = CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=f"{temp_dir}/README.md")
            commits.append(commit)

        API.create_commit(
            repo_id=args.repo_id,
            operations=commits,
            commit_message="Add `agent.pt` and tag the environment",
            commit_description="This commit adds a model that is compatible with the 🥇 [Open RL Leaderboard](https://huggingface.co/spaces/open-rl-leaderboard/leaderboard) 🥇.",
            repo_type="model",
            create_pr=True,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    make_and_upload_agent(args)
