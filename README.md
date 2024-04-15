# open-rl-leaderboard-utils

Utilities for pushing agents to the OpenRL Leaderboard.

## Stable-Baselines3

```bash
pip install stable-baselines3 sb3-contrib huggingface-hub gym==0.21 shimmy>=0.2.1
python sb3_push_agent.py --repo_id username/model_id
```

## CleanRL

```bash
pip install cleanrl
python cleanrl_push_agent.py --repo_id username/model_id
```
