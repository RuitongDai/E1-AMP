"""Circular replay buffer for AMP discriminator observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class AMPReplayBuffer:
  """Circular buffer with mini-batch generator for AMP discriminator training.

  Stores 3D tensors of shape (batch_size, disc_obs_steps, disc_obs_dim) and
  provides synchronized mini-batch iteration for training the discriminator.
  """

  def __init__(self, max_len: int, batch_size: int, device: str):
    if max_len < 1:
      raise ValueError(f"Buffer size must be >= 1, got {max_len}")
    self._batch_size = batch_size
    self._device = device
    self._max_len = torch.full((batch_size,), max_len, dtype=torch.int, device=device)
    self._num_pushes = torch.zeros(batch_size, dtype=torch.long, device=device)
    self._pointer: int = -1
    self._buffer: torch.Tensor | None = None

  @property
  def batch_size(self) -> int:
    return self._batch_size

  @property
  def device(self) -> str:
    return self._device

  @property
  def max_length(self) -> int:
    return int(self._max_len[0].item())

  @property
  def current_length(self) -> torch.Tensor:
    return torch.minimum(self._num_pushes, self._max_len)

  def reset(self, batch_ids: Sequence[int] | None = None) -> None:
    if batch_ids is None:
      batch_ids = slice(None)  # type: ignore[assignment]
    self._num_pushes[batch_ids] = 0
    if self._buffer is not None:
      self._buffer[:, batch_ids] = 0.0

  def append(self, data: torch.Tensor) -> None:
    if data.shape[0] != self._batch_size:
      raise ValueError(f"Expected batch dim {self._batch_size}, got {data.shape[0]}")
    data = data.to(self._device)
    if self._buffer is None:
      self._pointer = -1
      self._buffer = torch.empty(
        (self.max_length, *data.shape), dtype=data.dtype, device=self._device
      )
    self._pointer = (self._pointer + 1) % self.max_length
    self._buffer[self._pointer] = data
    is_first = self._num_pushes == 0
    if torch.any(is_first):
      self._buffer[:, is_first] = data[is_first]
    self._num_pushes += 1

  def mini_batch_generator(
    self,
    fetch_length: int,
    num_mini_batches: int,
    num_epochs: int = 8,
  ):
    if self._buffer is None or torch.any(self._num_pushes == 0):
      raise RuntimeError("Buffer is empty.")

    min_len = int(torch.min(self.current_length).item())
    if fetch_length > min_len:
      raise ValueError(f"fetch_length {fetch_length} > min current_length {min_len}")

    epoch_batch_size = self._batch_size * fetch_length
    mini_batch_size = epoch_batch_size // num_mini_batches

    total_combinations = int(self.current_length[0].item()) * self._batch_size
    linear_indices = torch.randperm(total_combinations, device=self._device)[
      :epoch_batch_size
    ]
    indices_0 = linear_indices // self._batch_size
    indices_1 = linear_indices % self._batch_size

    for _ in range(num_epochs):
      perm = torch.randperm(epoch_batch_size, requires_grad=False, device=self._device)
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        end = (i + 1) * mini_batch_size
        sel = perm[start:end]
        yield self._buffer[indices_0[sel], indices_1[sel]]
