# ==========================================
# 导入基础模块
# ==========================================
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ==========================================
# 导入任务相关模块
# ==========================================
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from . import mdp as my_mdp

# ==========================================
# 1. 场景配置 (Scene) - 彻底干净的舞台
# ==========================================
# 方块公共物理属性
COMMON_CUBE_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
@configclass
class FrankaVLACubeSceneCfg(InteractiveSceneCfg):
    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            color=(0.5, 0.5, 0.5),
            semantic_tags=[("class", "ground")]  # 移到了 spawn 内部
        ),
    )
    # 光照
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0)
    )
    # 桌子
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
            semantic_tags=[("class", "table")]  # 移到了 spawn 内部
        ),
    )
    
    # 机械臂
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    
    # 3个方块
    cube_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd", scale=(1.0, 1.0, 1.0), rigid_props=COMMON_CUBE_PROPS, semantic_tags=[("class", "cube_1")]),
    )
    cube_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.55, 0.05, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd", scale=(1.0, 1.0, 1.0), rigid_props=COMMON_CUBE_PROPS, semantic_tags=[("class", "cube_2")]),
    )
    cube_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.60, -0.1, 0.0203], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd", scale=(1.0, 1.0, 1.0), rigid_props=COMMON_CUBE_PROPS, semantic_tags=[("class", "cube_3")]),
    )

    # 末端执行器追踪
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # VLA 数据采集相机
    hand_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/hand_camera",
        update_period=0.1, height=224, width=224, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.02), rot=(0.13, 0.7, 0.7, 0.13), convention="opengl"),
    )
    ext_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/ext_camera",
        update_period=0.1, height=224, width=224, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=(1.38, -0.7, 1.05), rot=(0.78, 0.4, 0.2, 0.4), convention="opengl"),
    )

# ==========================================
# 2. MDP 配置 (状态、动作、奖励)
# ==========================================
@configclass
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose, mode="reset",
        params={"default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]},
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset, mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_cube_positions = EventTerm(
        func=franka_stack_events.randomize_object_pose, mode="reset",
        params={
            "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203), "yaw": (-1.0, 1, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2"), SceneEntityCfg("cube_3")],
        },
    )

@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0},
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        gripper_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"])})
        
        cube_1_pos = ObsTerm(func=my_mdp.cube_1_pos)
        cube_1_quat = ObsTerm(func=my_mdp.cube_1_quat)
        cube_2_pos = ObsTerm(func=my_mdp.cube_2_pos)
        cube_2_quat = ObsTerm(func=my_mdp.cube_2_quat)
        cube_3_pos = ObsTerm(func=my_mdp.cube_3_pos)
        cube_3_quat = ObsTerm(func=my_mdp.cube_3_quat)
        target_goal_id = ObsTerm(func=my_mdp.get_target_cube_id) 

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True 

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.1)
    penalty_wrong_grasp = RewTerm(func=my_mdp.grasped_wrong_cube, weight=-5.0)
    approach_target_cube = RewTerm(func=my_mdp.distance_to_target_cube, weight=1.0)
    grasp_target_cube = RewTerm(func=my_mdp.is_target_cube_grasped, weight=2.0)
    lift_target_cube = RewTerm(func=my_mdp.target_cube_height, weight=3.0)
    align_with_base_cube = RewTerm(func=my_mdp.distance_target_to_base_cube, weight=2.0)
    stack_success = RewTerm(func=my_mdp.is_stacked_successfully, weight=10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=my_mdp.is_stacked_successfully)
    cube_out_of_bounds = DoneTerm(func=my_mdp.any_cube_out_of_bounds)

# ==========================================
# 3. 主环境类 (继承最纯净的底层基类)
# ==========================================
@configclass
class VLADataCollectEnvCfg(ManagerBasedRLEnvCfg):
    # 挂载我们上面写的纯净配置
    scene: FrankaVLACubeSceneCfg = FrankaVLACubeSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # 初始化基础设置
        self.decimation = 2
        self.episode_length_s = 10.0
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        
        # 为机器人打上标签
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

@configclass
class VLADataCollectEnvCfg_PLAY(VLADataCollectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False