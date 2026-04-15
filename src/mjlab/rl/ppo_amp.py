"""PPO + AMP algorithm for rsl_rl 5.0.1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict

from mjlab.rl.amp_discriminator import (
  AMPDiscriminator,
  LossType,
  resolve_amp_config,
)
from mjlab.rl.amp_replay_buffer import AMPReplayBuffer


class PPOAMP(PPO):
  """PPO with Adversarial Motion Priors (AMP) discriminator."""

  def __init__(
    self,
    actor: MLPModel,
    critic: MLPModel,
    storage: RolloutStorage,
    disc_obs_buffer: AMPReplayBuffer,
    disc_demo_obs_buffer: AMPReplayBuffer,
    amp_cfg: dict,
    obs_groups: dict,
    **kwargs,
  ) -> None:
    # Remove amp_cfg from kwargs before passing to super
    super().__init__(actor, critic, storage, **kwargs)

    self.amp_cfg = amp_cfg
    self._obs_groups = obs_groups

    loss_type_str = self.amp_cfg.get("loss_type", "LSGAN")
    loss_map = {"GAN": LossType.GAN, "LSGAN": LossType.LSGAN, "WGAN": LossType.WGAN}
    if loss_type_str not in loss_map:
      raise ValueError(f"Unknown AMP loss type: {loss_type_str}")
    self.amp_loss_type = loss_map[loss_type_str]

    self.amp_discriminator = AMPDiscriminator(
      disc_obs_dim=self.amp_cfg["disc_obs_dim"],
      disc_obs_steps=self.amp_cfg["disc_obs_steps"],
      obs_groups=self._obs_groups,
      loss_type=self.amp_loss_type,
      device=self.device,
      **self.amp_cfg.get("amp_discriminator", {}),
    ).to(self.device)

    # Merged optimizer: policy (actor + critic) + discriminator in one optimizer
    # This follows TienKung's approach for joint backward pass
    self.optimizer = optim.Adam(
      [
        {"params": self.actor.parameters(), "name": "policy_actor"},
        {"params": self.critic.parameters(), "name": "policy_critic"},
        {
          "params": self.amp_discriminator.disc_trunk.parameters(),
          "weight_decay": self.amp_cfg.get("disc_trunk_weight_decay", 1e-4),
          "name": "amp_trunk",
        },
        {
          "params": self.amp_discriminator.disc_linear.parameters(),
          "weight_decay": self.amp_cfg.get("disc_linear_weight_decay", 1e-4),
          "name": "amp_head",
        },
      ],
      lr=self.learning_rate,
    )
    self.amploss_coef = 1.0

    self.disc_obs_buffer = disc_obs_buffer
    self.disc_demo_obs_buffer = disc_demo_obs_buffer

    # For logging
    self.style_rewards: torch.Tensor | None = None
    self.rewards_lerp: torch.Tensor | None = None

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
    disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(
      obs, flatten_history_dim=False
    )
    self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(
      disc_obs
    )
    self.rewards_lerp = self.amp_discriminator.lerp_reward(
      task_reward=rewards, style_reward=self.style_rewards
    )
    self.disc_obs_buffer.append(disc_obs)
    self.disc_demo_obs_buffer.append(disc_demo_obs)
    super().process_env_step(obs, self.rewards_lerp, dones, extras)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None
    mean_disc_loss = 0.0
    mean_disc_grad_penalty = 0.0
    mean_disc_score = 0.0
    mean_disc_demo_score = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    disc_obs_gen = self.disc_obs_buffer.mini_batch_generator(
      fetch_length=self.storage.num_transitions_per_env,
      num_mini_batches=self.num_mini_batches,
      num_epochs=self.num_learning_epochs,
    )
    disc_demo_obs_gen = self.disc_demo_obs_buffer.mini_batch_generator(
      fetch_length=self.storage.num_transitions_per_env,
      num_mini_batches=self.num_mini_batches,
      num_epochs=self.num_learning_epochs,
    )

    for batch, disc_obs_batch, disc_demo_obs_batch in zip(
      generator, disc_obs_gen, disc_demo_obs_gen
    ):
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      # Symmetric augmentation
      num_aug = 1
      if self.symmetry and self.symmetry["use_data_augmentation"]:
        data_augmentation_func = self.symmetry["data_augmentation_func"]
        batch.observations, batch.actions = data_augmentation_func(
          env=self.symmetry["_env"],
          obs=batch.observations,
          actions=batch.actions,
        )
        num_aug = int(batch.observations.batch_size[0] / original_batch_size)
        batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
        batch.values = batch.values.repeat(num_aug, 1)
        batch.advantages = batch.advantages.repeat(num_aug, 1)
        batch.returns = batch.returns.repeat(num_aug, 1)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      # Adaptive learning rate
      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      # Surrogate loss
      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      # Value loss
      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )

      # Symmetry loss
      if self.symmetry:
        if not self.symmetry["use_data_augmentation"]:
          data_augmentation_func = self.symmetry["data_augmentation_func"]
          batch.observations, _ = data_augmentation_func(
            obs=batch.observations, actions=None, env=self.symmetry["_env"]
          )
        mean_actions = self.actor(batch.observations.detach().clone())
        action_mean_orig = mean_actions[:original_batch_size]
        _, actions_mean_symm = data_augmentation_func(
          obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
        )
        mse_loss = torch.nn.MSELoss()
        symmetry_loss = mse_loss(
          mean_actions[original_batch_size:],
          actions_mean_symm.detach()[original_batch_size:],
        )
        if self.symmetry["use_mirror_loss"]:
          loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
        else:
          symmetry_loss = symmetry_loss.detach()

      # RND loss
      if self.rnd:
        with torch.no_grad():
          rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
          rnd_state = self.rnd.state_normalizer(rnd_state)
        predicted_embedding = self.rnd.predictor(rnd_state)
        target_embedding = self.rnd.target(rnd_state).detach()
        rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

      # AMP discriminator loss
      with torch.no_grad():
        disc_obs_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch)
        disc_demo_obs_normed = self.amp_discriminator.normalize_disc_obs(
          disc_demo_obs_batch
        )

      mb_size = disc_obs_normed.shape[0]
      disc_score = self.amp_discriminator(disc_obs_normed.reshape(mb_size, -1))
      disc_demo_score = self.amp_discriminator(
        disc_demo_obs_normed.reshape(mb_size, -1)
      )

      if self.amp_loss_type == LossType.GAN:
        bce = torch.nn.BCEWithLogitsLoss()
        policy_loss = bce(disc_score, torch.zeros_like(disc_score, device=self.device))
        demo_loss = bce(
          disc_demo_score,
          torch.ones_like(disc_demo_score, device=self.device),
        )
        disc_loss = 0.5 * (policy_loss + demo_loss)
      elif self.amp_loss_type == LossType.LSGAN:
        policy_loss = torch.nn.MSELoss()(
          disc_score, -1 * torch.ones_like(disc_score, device=self.device)
        )
        demo_loss = torch.nn.MSELoss()(
          disc_demo_score,
          torch.ones_like(disc_demo_score, device=self.device),
        )
        disc_loss = 0.5 * (policy_loss + demo_loss)
      elif self.amp_loss_type == LossType.WGAN:
        disc_loss = -torch.mean(disc_demo_score) + torch.mean(disc_score)
      else:
        raise ValueError(f"Unknown loss type: {self.amp_loss_type}")

      disc_grad_penalty = self.amp_discriminator.compute_grad_penalty(
        demo_data=disc_demo_obs_normed.reshape(mb_size, -1),
        scale=self.amp_cfg.get("grad_penalty_scale", 5.0),
      )

      # Joint backward: add AMP disc loss into the total loss
      loss += self.amploss_coef * disc_loss + self.amploss_coef * disc_grad_penalty

      # Single backward pass for policy + discriminator
      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd_optimizer:
        self.rnd_optimizer.step()
      self.amp_discriminator.update_normalization(disc_obs_batch)

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()
      mean_disc_loss += disc_loss.item()
      mean_disc_grad_penalty += disc_grad_penalty.item()
      mean_disc_score += disc_score.mean().item()
      mean_disc_demo_score += disc_demo_score.mean().item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    mean_disc_loss /= num_updates
    mean_disc_grad_penalty /= num_updates
    mean_disc_score /= num_updates
    mean_disc_demo_score /= num_updates

    self.storage.clear()

    loss_dict: dict[str, float] = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss  # type: ignore
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss  # type: ignore
    loss_dict["amp/disc_loss"] = mean_disc_loss
    loss_dict["amp/disc_grad_penalty"] = mean_disc_grad_penalty
    loss_dict["amp/disc_score"] = mean_disc_score
    loss_dict["amp/disc_demo_score"] = mean_disc_demo_score

    return loss_dict

  def train_mode(self) -> None:
    super().train_mode()
    self.amp_discriminator.train()
    self.amp_discriminator.disc_obs_normalizer.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    self.amp_discriminator.eval()
    self.amp_discriminator.disc_obs_normalizer.eval()

  def save(self) -> dict:
    saved_dict = super().save()
    saved_dict["amp_discriminator_state_dict"] = self.amp_discriminator.state_dict()
    saved_dict["amp_disc_normalizer_state_dict"] = (
      self.amp_discriminator.disc_obs_normalizer.state_dict()
    )
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    result = super().load(loaded_dict, load_cfg, strict)
    if "amp_discriminator_state_dict" in loaded_dict:
      self.amp_discriminator.load_state_dict(
        loaded_dict["amp_discriminator_state_dict"]
      )
    if "amp_disc_normalizer_state_dict" in loaded_dict:
      self.amp_discriminator.disc_obs_normalizer.load_state_dict(
        loaded_dict["amp_disc_normalizer_state_dict"]
      )
    return result

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> "PPOAMP":
    """Construct the PPOAMP algorithm with discriminator and replay buffers."""
    actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

    default_sets = ["actor", "critic"]
    if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
    cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

    actor: MLPModel = actor_class(
      obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
      cfg["critic"]["cnns"] = actor.cnns  # type: ignore
    critic: MLPModel = critic_class(
      obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
    ).to(device)
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )

    amp_cfg = cfg["algorithm"].pop("amp_cfg")
    disc_obs_buffer = AMPReplayBuffer(
      max_len=amp_cfg.get("disc_obs_buffer_size", 500),
      batch_size=env.num_envs,
      device=device,
    )
    disc_demo_obs_buffer = AMPReplayBuffer(
      max_len=amp_cfg.get("disc_obs_buffer_size", 500),
      batch_size=env.num_envs,
      device=device,
    )

    # Remove class_name before passing to constructor
    cfg["algorithm"].pop("class_name", None)

    alg = PPOAMP(
      actor,
      critic,
      storage,
      disc_obs_buffer,
      disc_demo_obs_buffer,
      amp_cfg=amp_cfg,
      obs_groups=cfg["obs_groups"],
      device=device,
      **cfg["algorithm"],
    )
    return alg
