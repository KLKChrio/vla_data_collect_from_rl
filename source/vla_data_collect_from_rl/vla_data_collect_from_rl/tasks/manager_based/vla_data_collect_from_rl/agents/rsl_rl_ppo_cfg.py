from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class FrankaStackPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 【核心修改 1】拉长步数。16 envs * 1024 steps = 16,384 样本/迭代
    num_steps_per_env = 1024 
    
    # 【核心修改 2】拉长总迭代次数。因为单次迭代的样本量少了，总迭代得补回来
    max_iterations = 6000  
    save_interval = 200    # 每 200 轮保存一次，避免硬盘爆满
    experiment_name = "franka_vla_stack_expert_16envs"
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        # 【核心修改 3】减小 mini-batch 数量，保证每个 batch 有足够的数据更新梯度
        num_mini_batches=2,  
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )