"""AMP Discriminator module for Adversarial Motion Priors."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from tensordict import TensorDict
from torch import autograd


class LossType(Enum):
  GAN = 0
  LSGAN = 1
  WGAN = 2


_ACT_MAP = {
  "elu": nn.ELU,
  "selu": nn.SELU,
  "relu": nn.ReLU,
  "crelu": nn.CELU,
  "lrelu": nn.LeakyReLU,
  "tanh": nn.Tanh,
  "sigmoid": nn.Sigmoid,
  "softplus": nn.Softplus,
  "gelu": nn.GELU,
  "swish": nn.SiLU,
  "mish": nn.Mish,
  "identity": nn.Identity,
}


def _resolve_activation(name: str) -> nn.Module:
  cls = _ACT_MAP.get(name.lower())
  if cls is None:
    raise ValueError(f"Unknown activation: {name}")
  return cls()


class AMPDiscriminator(nn.Module):
  def __init__(
    self,
    disc_obs_dim: int,
    disc_obs_steps: int,
    obs_groups: dict,
    loss_type: LossType = LossType.LSGAN,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "relu",
    amp_reward_coef: float = 1.0,
    task_style_lerp: float = 0.0,
    lsgan_reward_scale: float = 0.25,
    device: str = "cpu",
  ):
    super().__init__()

    self.input_dim = disc_obs_dim * disc_obs_steps
    self.disc_obs_dim = disc_obs_dim
    self.disc_obs_steps = disc_obs_steps
    self.obs_groups = obs_groups
    act = _resolve_activation(activation)
    self.amp_reward_coef = amp_reward_coef
    self.task_style_lerp = task_style_lerp
    self.lsgan_reward_scale = lsgan_reward_scale
    self.device = device
    self.loss_type = loss_type

    self.disc_obs_normalizer = EmpiricalNormalization(
      shape=self.disc_obs_dim, until=int(1e8)
    ).to(device)

    disc_layers: list[nn.Module] = []
    curr_in_dim = self.input_dim
    for hidden_dim in hidden_dims:
      disc_layers.append(nn.Linear(curr_in_dim, hidden_dim))
      disc_layers.append(act)
      curr_in_dim = hidden_dim
    self.disc_trunk = nn.Sequential(*disc_layers)
    self.disc_linear = nn.Linear(hidden_dims[-1], 1)

    if self.loss_type == LossType.WGAN:
      self.disc_output_normalizer = EmpiricalNormalization(shape=1, until=int(1e8)).to(
        device
      )
    else:
      self.disc_output_normalizer = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.disc_trunk(x)
    return self.disc_linear(h)

  def get_disc_obs(
    self, obs: TensorDict, flatten_history_dim: bool = False
  ) -> torch.Tensor:
    disc_obs_list = []
    for obs_group in self.obs_groups["discriminator"]:
      if obs_group not in obs:
        raise ValueError(f"Observation group '{obs_group}' not found in observations.")
      obs_tensor = obs[obs_group]
      assert len(obs_tensor.shape) == 3
      num_envs, history_length, obs_dim = obs_tensor.shape
      assert history_length == self.disc_obs_steps
      disc_obs_list.append(obs_tensor)
    disc_obs = torch.cat(disc_obs_list, dim=-1)
    if flatten_history_dim:
      disc_obs = disc_obs.view(num_envs, -1)  # noqa: F821
    return disc_obs

  def get_disc_demo_obs(
    self, obs: TensorDict, flatten_history_dim: bool = False
  ) -> torch.Tensor:
    disc_demo_obs_list = []
    for obs_group in self.obs_groups["discriminator_demonstration"]:
      if obs_group not in obs:
        raise ValueError(f"Observation group '{obs_group}' not found in observations.")
      obs_tensor = obs[obs_group]
      assert len(obs_tensor.shape) == 3
      num_envs, history_length, obs_dim = obs_tensor.shape
      assert history_length == self.disc_obs_steps
      disc_demo_obs_list.append(obs_tensor)
    disc_demo_obs = torch.cat(disc_demo_obs_list, dim=-1)
    if flatten_history_dim:
      disc_demo_obs = disc_demo_obs.reshape(num_envs, -1)  # noqa: F821
    return disc_demo_obs

  def normalize_disc_obs(self, disc_obs: torch.Tensor) -> torch.Tensor:
    disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
    normed = self.disc_obs_normalizer(disc_obs_reshaped)
    return normed.reshape(-1, self.disc_obs_steps, self.disc_obs_dim)

  def update_normalization(self, disc_obs: torch.Tensor) -> None:
    disc_obs_reshaped = disc_obs.reshape(-1, self.disc_obs_dim)
    self.disc_obs_normalizer.update(disc_obs_reshaped)

  def compute_grad_penalty(
    self, demo_data: torch.Tensor, scale: float = 10.0
  ) -> torch.Tensor:
    demo_data_copy = demo_data.clone().detach().requires_grad_(True)
    disc = self.forward(demo_data_copy)
    ones = torch.ones_like(disc, device=demo_data_copy.device)
    grad = autograd.grad(
      outputs=disc,
      inputs=demo_data_copy,
      grad_outputs=ones,
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return scale * (grad.norm(2, dim=1) - 0).pow(2).mean()

  def predict_style_reward(
    self, disc_obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    was_training = self.training
    with torch.no_grad():
      self.eval()
      disc_obs_reshaped = disc_obs.view(-1, self.disc_obs_dim)
      normed = self.disc_obs_normalizer(disc_obs_reshaped)
      normed = normed.view(-1, self.disc_obs_steps * self.disc_obs_dim)
      disc_score = self.forward(normed)

      if self.loss_type == LossType.GAN:
        prob = 1.0 / (1.0 + torch.exp(-disc_score))
        rew = -torch.log(
          torch.maximum(1 - prob, torch.tensor(1e-6, device=self.device))
        )
      elif self.loss_type == LossType.LSGAN:
        rew = torch.clamp(
          1 - self.lsgan_reward_scale * torch.square(disc_score - 1), min=0
        )
      elif self.loss_type == LossType.WGAN:
        rew = self.disc_output_normalizer(disc_score)
      else:
        raise ValueError(f"Unknown AMP loss type: {self.loss_type}")

      style_reward = self.amp_reward_coef * rew

      if was_training:
        self.train()
        if self.loss_type == LossType.WGAN:
          self.disc_output_normalizer.update(disc_score)

    return style_reward.squeeze(-1), disc_score.squeeze(-1)

  def lerp_reward(
    self, task_reward: torch.Tensor, style_reward: torch.Tensor
  ) -> torch.Tensor:
    return (
      self.task_style_lerp * task_reward + (1.0 - self.task_style_lerp) * style_reward
    )


def resolve_amp_config(
  alg_cfg: dict, obs: TensorDict, obs_groups: dict, env: VecEnv
) -> dict:
  if "amp_cfg" not in alg_cfg or alg_cfg["amp_cfg"] is None:
    raise ValueError("AMP configuration is missing or None.")

  disc_obs_dim = 0
  disc_obs_steps = -1

  if (
    "discriminator" not in obs_groups or "discriminator_demonstration" not in obs_groups
  ):
    raise ValueError(
      "AMP requires 'discriminator' and 'discriminator_demonstration' obs groups."
    )

  for obs_group in obs_groups["discriminator"]:
    if obs_group not in obs:
      raise ValueError(f"Obs group '{obs_group}' not found in observations.")
    obs_tensor = obs[obs_group]
    assert len(obs_tensor.shape) == 3
    if disc_obs_steps == -1:
      disc_obs_steps = obs_tensor.shape[1]
    else:
      assert disc_obs_steps == obs_tensor.shape[1]
    disc_obs_dim += obs_tensor.shape[-1]

  disc_demo_obs_dim = 0
  for obs_group in obs_groups["discriminator_demonstration"]:
    if obs_group not in obs:
      raise ValueError(f"Obs group '{obs_group}' not found in observations.")
    obs_tensor = obs[obs_group]
    assert len(obs_tensor.shape) == 3
    assert disc_obs_steps == obs_tensor.shape[1]
    disc_demo_obs_dim += obs_tensor.shape[-1]

  assert disc_demo_obs_dim == disc_obs_dim

  alg_cfg["amp_cfg"]["disc_obs_steps"] = disc_obs_steps
  alg_cfg["amp_cfg"]["disc_obs_dim"] = disc_obs_dim

  return alg_cfg
