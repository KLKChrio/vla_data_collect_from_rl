# tasks/manager_based/vla_data_collect_from_rl/__init__.py
import gymnasium as gym

from . import vla_data_collect_from_rl_env_cfg
from .agents import rsl_rl_ppo_cfg

# 注册用于训练的环境
gym.register(
    id="Franka-VLA-Stack-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vla_data_collect_from_rl_env_cfg.VLADataCollectEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaStackPPORunnerCfg,
    },
)

# 注册用于 Play / 录制数据的环境 (数量少，无随机化)
gym.register(
    id="Franka-VLA-Stack-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vla_data_collect_from_rl_env_cfg.VLADataCollectEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.FrankaStackPPORunnerCfg,
    },
)