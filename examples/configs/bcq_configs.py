from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field

@dataclass
class BCQTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "BCQ"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True

    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0

    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cpu"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.001
    critic_lr: float = 0.001
    vae_lr: float = 0.001
    phi: float = 0.05
    lmbda: float = 0.75
    beta: float = 0.5
    cost_limit: int = 10
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8

    # model params
    a_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[256, 256], is_mutable=True)
    vae_hidden_sizes: int = 400
    sample_action_num: int = 10
    gamma: float = 0.99
    tau: float = 0.005
    num_q: int = 2
    num_qc: int = 2
    PID: List[float] = field(default=[0.1, 0.003, 0.001], is_mutable=True)

    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class BCQCarCircleConfig(BCQTrainConfig):
    pass


@dataclass
class BCQAntRunConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class BCQDroneRunConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BCQDroneCircleConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BCQCarRunConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BCQAntCircleConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class BCQBallRunConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class BCQBallCircleConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class BCQCarButton1Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQCarButton2Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQCarCircle1Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQCarCircle2Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQCarGoal1Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQCarGoal2Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQCarPush1Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQCarPush2Config(BCQTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQPointButton1Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQPointButton2Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQPointCircle1Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQPointCircle2Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class BCQPointGoal1Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQPointGoal2Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQPointPush1Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQPointPush2Config(BCQTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class BCQAntVelocityConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQHalfCheetahVelocityConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQHopperVelocityConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQSwimmerVelocityConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQWalker2dVelocityConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class BCQEasySparseConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQEasyMeanConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQEasyDenseConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQMediumSparseConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQMediumMeanConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQMediumDenseConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQHardSparseConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQHardMeanConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class BCQHardDenseConfig(BCQTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


BCQ_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCQCarCircleConfig,
    "OfflineAntRun-v0": BCQAntRunConfig,
    "OfflineDroneRun-v0": BCQDroneRunConfig,
    "OfflineDroneCircle-v0": BCQDroneCircleConfig,
    "OfflineCarRun-v0": BCQCarRunConfig,
    "OfflineAntCircle-v0": BCQAntCircleConfig,
    "OfflineBallCircle-v0": BCQBallCircleConfig,
    "OfflineBallRun-v0": BCQBallRunConfig,
    # safety_gymnasium: car
    "OfflineCarButton1Gymnasium-v0": BCQCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": BCQCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": BCQCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": BCQCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": BCQCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": BCQCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": BCQCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": BCQCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": BCQPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": BCQPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": BCQPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": BCQPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": BCQPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": BCQPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": BCQPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": BCQPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": BCQAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": BCQHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": BCQHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": BCQSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": BCQWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": BCQEasySparseConfig,
    "OfflineMetadrive-easymean-v0": BCQEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": BCQEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": BCQMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": BCQMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": BCQMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": BCQHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": BCQHardMeanConfig,
    "OfflineMetadrive-harddense-v0": BCQHardDenseConfig
}