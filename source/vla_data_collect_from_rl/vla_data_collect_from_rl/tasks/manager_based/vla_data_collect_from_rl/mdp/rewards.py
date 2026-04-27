from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)

# --- 辅助函数：获取目标和基座的坐标 ---
def _get_target_and_base_pos(env: ManagerBasedRLEnv):
    """提取当前所有环境中，目标方块和目标基座的位置"""
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]
    
    all_cubes_pos = torch.stack([
        cube_1.data.root_pos_w, 
        cube_2.data.root_pos_w, 
        cube_3.data.root_pos_w
    ], dim=1)

    # 假设目标逻辑：总是把 Cube 0 (蓝) 放到 Cube 1 (红) 上
    target_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    base_idx = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
    env_indices = torch.arange(env.num_envs, device=env.device)
    target_cube_pos = all_cubes_pos[env_indices, target_idx]
    base_cube_pos = all_cubes_pos[env_indices, base_idx]

    return target_cube_pos, base_cube_pos

# --- 核心奖励函数 ---

def distance_to_target_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """奖励 2：接近目标 cube"""
    ee_frame = env.scene["ee_frame"] 
    ee_pos = ee_frame.data.target_pos_w[:, 0, :] 

    target_cube_pos, _ = _get_target_and_base_pos(env)
    
    distance = torch.norm(target_cube_pos - ee_pos, dim=-1)
    reward = 1.0 / (1.0 + 10.0 * distance)
    return reward

def is_target_cube_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """奖励 3.1：是否成功抓取了目标方块"""
    target_cube_pos, _ = _get_target_and_base_pos(env)
    
    ee_frame = env.scene["ee_frame"]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    
    distance = torch.norm(target_cube_pos - ee_pos, dim=-1)
    is_close = distance < 0.04 
    
    # 结合 Z 轴高度判断 (初始高度0.0203，大于0.03视为真正离地，防止只把手搭在方块上)
    is_lifted = target_cube_pos[:, 2] > 0.03
    
    return (is_close & is_lifted).float()

def target_cube_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """奖励 3.2：举起目标方块的奖励 (鼓励举高)"""
    target_cube_pos, _ = _get_target_and_base_pos(env)
    
    # 计算相对桌面的离地高度
    relative_height = target_cube_pos[:, 2] - 0.0203
    clamped_height = torch.clamp(relative_height, min=0.0, max=0.3)
    
    return clamped_height

def distance_target_to_base_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """奖励 4：放到指定 cube 上方 (目标方块和基座方块的对齐距离)"""
    target_cube_pos, base_cube_pos = _get_target_and_base_pos(env)
    
    xy_distance = torch.norm(target_cube_pos[:, :2] - base_cube_pos[:, :2], dim=-1)
    is_grasped = is_target_cube_grasped(env)
    
    reward = 1.0 / (1.0 + 10.0 * xy_distance)
    return reward * is_grasped

def is_stacked_successfully(env: ManagerBasedRLEnv) -> torch.Tensor:
    """奖励 5 & 终止条件：成功堆叠"""
    target_cube_pos, base_cube_pos = _get_target_and_base_pos(env)
    
    # 条件 1：XY 平面对齐
    xy_distance = torch.norm(target_cube_pos[:, :2] - base_cube_pos[:, :2], dim=-1)
    is_aligned = xy_distance < 0.04 
    
    # 条件 2：Z 轴高度差 (由于边长已知，堆叠后目标方块的质心高度应比基座高一个边长的距离)
    z_diff = target_cube_pos[:, 2] - base_cube_pos[:, 2]
    is_on_top = (z_diff > 0.03) & (z_diff < 0.06)
    
    # 条件 3：【动态提取】目标方块线速度趋于静止
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]
    all_cubes_vel = torch.stack([
        cube_1.data.root_lin_vel_w, 
        cube_2.data.root_lin_vel_w, 
        cube_3.data.root_lin_vel_w
    ], dim=1)
    
    target_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env_indices = torch.arange(env.num_envs, device=env.device)
    target_cube_vel = all_cubes_vel[env_indices, target_idx]
    
    vel = torch.norm(target_cube_vel, dim=-1)
    is_static = vel < 0.05
    
    return (is_aligned & is_on_top & is_static)

def any_cube_out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    """终止条件：方块掉出桌子"""
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]
    
    # 【修复】检查所有三个方块，如果任何一个掉下去 (< -0.1) 就终止
    all_cubes_z = torch.stack([
        cube_1.data.root_pos_w[:, 2], 
        cube_2.data.root_pos_w[:, 2], 
        cube_3.data.root_pos_w[:, 2]
    ], dim=1)
    
    out_of_bounds = (all_cubes_z < -0.1).any(dim=1)
    return out_of_bounds

def grasped_wrong_cube(env: ManagerBasedRLEnv) -> torch.Tensor:
    """惩罚 1：抓错了方块"""
    cube_1: RigidObject = env.scene["cube_1"]
    cube_2: RigidObject = env.scene["cube_2"]
    cube_3: RigidObject = env.scene["cube_3"]
    
    all_cubes_pos = torch.stack([
        cube_1.data.root_pos_w, 
        cube_2.data.root_pos_w, 
        cube_3.data.root_pos_w
    ], dim=1)
    
    ee_frame = env.scene["ee_frame"] 
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    
    target_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # 1. 计算所有方块到末端的距离
    dist_to_all = torch.norm(all_cubes_pos - ee_pos.unsqueeze(1), dim=-1)
    is_close = dist_to_all < 0.04
    
    # 2. 判断所有方块是否被举起 (初始高度0.0203，大于0.03视为离地)
    is_lifted = all_cubes_pos[:, :, 2] > 0.03
    
    # 3. 抓取条件 = 靠近 + 离地
    is_grasped = is_close & is_lifted
    
    # 4. 掩码生成：排除掉目标方块，我们只惩罚非目标方块被抓取
    env_indices = torch.arange(env.num_envs, device=env.device)
    mask = torch.ones((env.num_envs, 3), dtype=torch.bool, device=env.device)
    mask[env_indices, target_idx] = False 
    
    # 5. 取交集：如果有任何非目标方块被抓起，则触发惩罚
    wrong_grasped = (is_grasped & mask).any(dim=1)
    
    return wrong_grasped