from mjlab.rl.amp_runner import MjlabAmpRunner
from mjlab.tasks.amp.amp_env import AmpEnv
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_go1_amp_env_cfg
from .rl_cfg import unitree_go1_amp_runner_cfg

register_mjlab_task(
  task_id="Mjlab-AMP-Flat-Unitree-Go1",
  env_cfg=unitree_go1_amp_env_cfg(),
  play_env_cfg=unitree_go1_amp_env_cfg(play=True),
  rl_cfg=unitree_go1_amp_runner_cfg(),
  runner_cls=MjlabAmpRunner,
  env_class=AmpEnv,
)
