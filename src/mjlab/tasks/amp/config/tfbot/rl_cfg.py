"""RL configuration for TFBOT AMP task."""

from mjlab.rl import (
  RslRlAmpCfg,
  RslRlAmpDiscriminatorCfg,
  RslRlAmpOnPolicyRunnerCfg,
  RslRlModelCfg,
  RslRlPpoAmpAlgorithmCfg,
)
from mjlab.tasks.amp.config.tfbot.symmetry import compute_symmetric_states


def tfbot_amp_runner_cfg() -> RslRlAmpOnPolicyRunnerCfg:
  """Create RL runner configuration for TFBOT AMP task."""
  return RslRlAmpOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      symmetry_cfg={
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "mirror_loss_coeff": 0.2,
        "data_augmentation_func": compute_symmetric_states,
      },
      amp_cfg=RslRlAmpCfg(
        disc_obs_buffer_size=100,
        disc_trunk_weight_decay=1e-3,
        disc_linear_weight_decay=1e-1,
        grad_penalty_scale=10.0,
        amp_discriminator=RslRlAmpDiscriminatorCfg(
          hidden_dims=(1024, 512, 256),
          activation="relu",
          amp_reward_coef=0.3,
          task_style_lerp=0.7,
        ),
        loss_type="LSGAN",
      ),
    ),
    obs_groups={
      "policy": ["actor"],
      "critic": ["critic"],
      "discriminator": ["disc"],
      "discriminator_demonstration": ["disc_demo"],
    },
    experiment_name="tfbot_amp",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
