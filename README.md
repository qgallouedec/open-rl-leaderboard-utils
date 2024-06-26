# open-rl-leaderboard-utils

Utilities for pushing agents to the OpenRL Leaderboard.

## Demo

```bash
pip install torch huggingface-hub gymnasium
python demo.py --hf_user my_username
```

## Stable-Baselines3

```bash
pip install stable-baselines3 rl-zoo3 huggingface_sb3 gym==0.21 shimmy>=0.2.1
python sb3_push_agent.py --repo_id username/model_id
```

## CleanRL

```bash
pip install cleanrl
python cleanrl_push_agent.py --repo_id username/model_id
```

## Sample Factory

### Atari

```bash
pip install sample_factory gymnasium[accept-rom-license,atari]
python sf_push_agent_atari.py --repo_id username/model_id
```
