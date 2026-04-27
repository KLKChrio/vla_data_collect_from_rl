import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject

def get_target_cube_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """目标指令 (目前固定为 0，即蓝色方块)"""
    num_envs = env.num_envs
    target_idx = torch.zeros(num_envs, dtype=torch.long, device=env.device) 
    return torch.nn.functional.one_hot(target_idx, num_classes=3).float()

# --- 上帝视角：获取方块的位置和旋转姿态 ---
def cube_1_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube: RigidObject = env.scene["cube_1"]
    # 减去环境原点坐标，将其转换为相对于每个 env 局部的坐标 (极度关键，否则 PPO 无法泛化)
    return cube.data.root_pos_w - env.scene.env_origins

def cube_1_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube: RigidObject = env.scene["cube_1"]
    return cube.data.root_quat_w

def cube_2_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube: RigidObject = env.scene["cube_2"]
    return cube.data.root_pos_w - env.scene.env_origins

def cube_2_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube: RigidObject = env.scene["cube_2"]
    return cube.data.root_quat_w

def cube_3_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube: RigidObject = env.scene["cube_3"]
    return cube.data.root_pos_w - env.scene.env_origins

def cube_3_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    cube: RigidObject = env.scene["cube_3"]
    return cube.data.root_quat_w