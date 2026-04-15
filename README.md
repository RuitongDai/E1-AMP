# T1 Humanoid AMP Walking — 完整工程

本项目是一个 **完全自包含的可安装 Python 工程**，包含 Booster T1 (23-DOF) 人形机器人基于 **AMP (Adversarial Motion Priors)** 的行走策略全流程：从人体动作捕捉数据重定向、强化学习训练、策略回放评估，到 MuJoCo 原生仿真验证 (sim2sim) 与键盘控制。

**收到本工程后，新电脑只需 `pip install -e .` 即可安装所有训练框架和依赖。**

**已包含的核心组件（全部内置，无需外部下载）：**

| 组件 | 说明 | 位置 |
|------|------|------|
| **mjlab** (训练框架) | MJWarp GPU 并行仿真 + AMP/PPO 训练完整源码 | `src/mjlab/` |
| **GMR** (动作重定向) | SMPL-X → T1 逆运动学重定向完整源码 | `general_motion_retargeting/` |
| **T1 机器人模型** | 23-DOF MJCF XML + 65 STL 网格 | `src/mjlab/asset_zoo/robots/booster_t1/` |
| **AMP 参考运动** | 8 组 ACCAD 行走动捕 (.pkl) | `src/mjlab/tasks/amp/data/t1_motions/` |
| **GMR 机器人资产** | T1 URDF + mesh（重定向用） | `gmr_assets/booster_t1/` |
| **预训练检查点** | model_26250.pt | `checkpoints/` |
| **Sim2Sim 脚本** | 纯 MuJoCo 验证 + 键盘控制 (568行) | `scripts/sim2sim_t1_mjlab.py` |

> `pip install -e .` 安装后会自动拉取的第三方依赖：MuJoCo (≥3.6.0)、MuJoCo-Warp (≥3.6.0)、PyTorch (≥2.7.0)、rsl-rl-lib (5.0.1)、Warp-lang (≥1.12.0) 等。

---

## 目录结构

```
T1-AMP-Walking/
├── pyproject.toml                     # pip install -e . 入口（安装 mjlab + GMR + 所有依赖）
├── build_package.sh                   # 打包构建脚本（仅开发者用）
├── environment.yml                    # Conda 环境定义（备选安装方式）
├── environment-retarget.yml           # Conda 环境定义（GMR 动作重定向，可选）
├── README.md                          # 本文档
│
├── src/mjlab/                         # ★ mjlab 训练框架完整源码
│   ├── __init__.py                    #   包初始化 + MJLAB_SRC_PATH
│   ├── actuator/                      #   执行器模型 (DcMotor, PD, etc.)
│   ├── asset_zoo/                     #   机器人资产
│   │   └── robots/booster_t1/         #   T1 机器人 (meshes/ + xmls/ + constants)
│   ├── envs/                          #   环境基础设施 + MDP
│   ├── managers/                      #   RL 管理器 (obs, action, reward, etc.)
│   ├── rl/                            #   RL 训练代码 (PPO + AMP判别器 + runner)
│   ├── scene/                         #   场景管理
│   ├── scripts/                       #   CLI 入口 (train.py, play.py)
│   ├── sensor/                        #   传感器 (contact, camera)
│   ├── sim/                           #   仿真核心
│   ├── tasks/                         #   任务定义
│   │   └── amp/                       #   ★ AMP 任务
│   │       ├── config/t1/             #     T1 训练配置 (env_cfgs, rl_cfg, symmetry)
│   │       ├── data/t1_motions/       #     8 组参考运动数据 (.pkl)
│   │       ├── managers/              #     AMP 动画/运动数据管理器
│   │       └── mdp/                   #     AMP 观测/奖励/终止
│   ├── terrains/                      #   地形生成
│   ├── utils/                         #   工具函数
│   └── viewer/                        #   可视化
│
├── general_motion_retargeting/        # ★ GMR 动作重定向完整源码
│   ├── __init__.py
│   ├── motion_retarget.py             #   主重定向引擎 (IK 求解)
│   ├── kinematics_model.py            #   运动学模型
│   ├── ik_configs/                    #   30 个 IK 映射配置 (含 smplx_to_t1.json)
│   └── utils/                         #   工具 + vendor 库
│
├── gmr_assets/                        # GMR 机器人资产
│   └── booster_t1/                    #   T1 URDF + mesh（重定向用）
│
├── scripts/
│   └── sim2sim_t1_mjlab.py            # Sim2Sim + 键盘控制脚本
│
├── retarget/                          # 重定向辅助脚本
│   ├── ik_configs/smplx_to_t1.json
│   └── scripts/
│       ├── retarget_t1_walking.sh     # 批量重定向 8 组动捕数据
│       └── convert_gmr_to_amp.py      # GMR → AMP 格式转换
│
└── checkpoints/
    └── model_26250.pt                 # 预训练策略检查点
```

---

## 1. 环境配置与安装

### 1.1 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Linux (Ubuntu 20.04+) |
| **GPU** | NVIDIA GPU (建议 RTX 3090 或更高) |
| **CUDA** | 12.1+ (推荐 12.8) |
| **Python** | 3.10 / 3.11 / 3.12 / 3.13（推荐 **3.11**） |
| **磁盘** | ≥ 10 GB（含 PyTorch + MuJoCo + 训练日志） |

### 1.2 方法一：使用 UV 安装（推荐，速度快）

[uv](https://docs.astral.sh/uv/) 是一个极速 Python 包管理工具，安装速度远快于 pip/conda。

```bash
# 1. 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 进入本项目根目录
cd T1-AMP-Walking

# 3. 创建虚拟环境 + 安装本项目及所有依赖
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# 这一步会自动安装本项目内置的 mjlab 训练框架 + GMR 动作重定向 + 以下第三方依赖：
#   • torch >= 2.7.0           (神经网络训练)
#   • mujoco >= 3.6.0          (物理引擎)
#   • mujoco-warp >= 3.6.0     (GPU 并行仿真)
#   • warp-lang >= 1.12.0      (NVIDIA Warp 加速)
#   • rsl-rl-lib == 5.0.1      (RSL 强化学习库，含 PPO + AMP 判别器)
#   • numpy, tensorboard, wandb 等

# 4. 验证安装
python -c "import mjlab; import mujoco; import torch; print('mjlab + mujoco + torch OK')"
python -c "import general_motion_retargeting; print('GMR OK')"
```

### 1.3 方法二：使用 Conda + pip 安装

```bash
# 1. 创建 conda 环境（Python 3.11）
conda create -n t1-amp python=3.11 -y
conda activate t1-amp

# 2. 安装 PyTorch（根据 CUDA 版本选择，二选一）
# CUDA 12.8:
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CUDA 12.1:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. 进入本项目根目录，安装本项目
cd T1-AMP-Walking
pip install -e .
# 这一步等同于 uv 安装——会安装项目内置的 mjlab + GMR 源码包，
# 并自动拉取 mujoco, mujoco-warp, rsl-rl-lib, warp-lang 等全部依赖。

# 4. 验证安装
python -c "import mjlab; import mujoco; import torch; print('mjlab + mujoco + torch OK')"
python -c "import general_motion_retargeting; print('GMR OK')"
```

> **备选方式：** 使用 `environment.yml` 一键创建 conda 环境并安装：
> ```bash
> cd T1-AMP-Walking
> conda env create -f environment.yml
> conda activate t1-amp
> ```

### 1.4 安装 GMR 额外依赖（可选，仅重定向需要）

如需从原始 SMPL-X 动捕数据重新生成参考运动（本项目已包含 8 组转换好的 .pkl，**通常不需要**）：

```bash
# 在已有环境中追加 GMR 重定向依赖
pip install -e ".[retarget]"
```

### 1.5 安装后目录说明

安装完成后，`import mjlab` 和 `import general_motion_retargeting` 均可用：

| 你得到的 | 功能 |
|----------|------|
| `mjlab.scripts.train` | 训练入口（`python -m mjlab.scripts.train Mjlab-AMP-Flat-T1`） |
| `mjlab.scripts.play` | 回放入口（`python -m mjlab.scripts.play Mjlab-AMP-Flat-T1`） |
| `mjlab.rl` | RL 训练代码（PPO + AMP 判别器 + runner） |
| `mjlab.tasks.amp` | AMP 任务定义 + T1 配置 + 运动数据 |
| `mjlab.actuator` | 执行器模型（DcMotor 等） |
| `mjlab.asset_zoo.robots.booster_t1` | T1 机器人定义 + XML + 网格 |
| `general_motion_retargeting` | GMR 动作重定向引擎 |

### 1.6 核心依赖速查表

| 包名 | 版本 | 安装方式 | 用途 |
|------|------|----------|------|
| `mjlab` | 1.2.0 | **内置** (`src/mjlab/`) | 训练框架 (MJWarp GPU 仿真) |
| `general_motion_retargeting` | 0.2.0 | **内置** (`general_motion_retargeting/`) | 动作重定向 (IK 求解) |
| `torch` | ≥ 2.7.0 | pip 自动安装 | 神经网络 (Actor/Critic/Discriminator) |
| `mujoco` | ≥ 3.6.0 | pip 自动安装 | 物理引擎 (训练 + sim2sim) |
| `mujoco-warp` | ≥ 3.6.0 | pip 自动安装 | GPU 批量并行仿真 (4096 环境) |
| `warp-lang` | ≥ 1.12.0 | pip 自动安装 | NVIDIA Warp 加速 |
| `rsl-rl-lib` | 5.0.1 | pip 自动安装 | RSL RL 库 (PPO + AMP) |

---

## 2. 全流程管线

### 阶段 A：动作重定向（Motion Retargeting）

> **注：** 本项目已包含转换后的 8 组参考运动（位于 `src/mjlab/tasks/amp/data/t1_motions/`），**可直接跳到阶段 B**。此步骤仅在需要从原始动捕数据重新生成时执行。

将 SMPL-X 格式的人体动捕数据（ACCAD 数据集）重定向到 T1 机器人关节空间：

```bash
cd T1-AMP-Walking

# 安装 GMR 额外依赖（如未安装）
pip install -e ".[retarget]"

# 批量重定向 8 组行走动作
bash retarget/scripts/retarget_t1_walking.sh

# 转换 GMR .pkl → mjlab AMP .pkl（四元数 xyzw→wxyz，FK 计算关键身体位置等）
python retarget/scripts/convert_gmr_to_amp.py \
  --input t1_retarget_output/female_walk1.pkl \
  --output src/mjlab/tasks/amp/data/t1_motions/female_walk1.pkl \
  --robot booster_t1
```

### 阶段 B：AMP 训练

```bash
cd T1-AMP-Walking

# 启动训练（4096 并行环境，50000 迭代）
python -m mjlab.scripts.train Mjlab-AMP-Flat-T1

# 训练日志保存于：logs/rsl_rl/t1_amp/<timestamp>/
# 检查点每 50 次迭代保存一次
```

如需离线模式（不上传 wandb）：
```bash
WANDB_MODE=offline python -m mjlab.scripts.train Mjlab-AMP-Flat-T1
```

**训练参数概要：**

| 参数 | 值 |
|------|-----|
| 并行环境数 | 4096 |
| 最大迭代次数 | 50,000 |
| 仿真时间步 | 0.005s |
| Decimation | 4 (控制频率 50Hz) |
| 回合长度 | 20s |
| 保存间隔 | 50 iterations |
| Actor 网络 | MLP (512, 256, 128) + ELU |
| Critic 网络 | MLP (512, 256, 128) + ELU |
| 判别器网络 | MLP (1024, 512, 256) + ReLU |

**训练提示：**
- AMP 训练前 ~5000 次迭代机器人可能原地站立，这是正常行为
- 约 10000–15000 迭代后开始出现稳定行走
- 完整训练约需 25000–50000 迭代

### 阶段 C：策略回放（Play）

```bash
cd T1-AMP-Walking

# 使用内置预训练检查点回放
python -m mjlab.scripts.play Mjlab-AMP-Flat-T1 \
  --checkpoint-file checkpoints/model_26250.pt

# 或使用自己训练的检查点
python -m mjlab.scripts.play Mjlab-AMP-Flat-T1 \
  --checkpoint-file logs/rsl_rl/t1_amp/<timestamp>/model_XXXXX.pt
```

回放模式固定速度指令 vx=0.3 m/s，无限回合长度。

### 阶段 D：Sim2Sim + 键盘控制

将训练好的策略在纯 MuJoCo 原生仿真中运行验证：

```bash
cd T1-AMP-Walking

# 带可视化 + 键盘控制（使用内置检查点）
python scripts/sim2sim_t1_mjlab.py \
  --checkpoint checkpoints/model_26250.pt \
  --keyboard

# 固定速度，不启用键盘（前进 0.3 m/s）
python scripts/sim2sim_t1_mjlab.py \
  --checkpoint checkpoints/model_26250.pt \
  --vx 0.3

# 无头模式（用于自动验证，运行 10 秒）
python scripts/sim2sim_t1_mjlab.py \
  --checkpoint checkpoints/model_26250.pt \
  --headless --duration 10 --vx 0.3

# 使用自己训练的检查点
python scripts/sim2sim_t1_mjlab.py \
  --checkpoint logs/rsl_rl/t1_amp/<timestamp>/model_XXXXX.pt \
  --keyboard
```

**键盘控制（在终端窗口中按键）：**

| 按键 | 功能 |
|------|------|
| `W` / `↑` | 前进 (vx +0.1 m/s) |
| `S` / `↓` | 后退 (vx -0.1 m/s) |
| `A` / `←` | 左移 (vy +0.05 m/s) |
| `D` / `→` | 右移 (vy -0.05 m/s) |
| `Q` | 左转 (wyaw +0.1 rad/s) |
| `E` | 右转 (wyaw -0.1 rad/s) |
| `Space` | 全停 (归零所有速度指令) |
| `Ctrl+C` | 退出 |

**速度范围限制：** vx ∈ [-0.5, 0.7], vy ∈ [-0.2, 0.2], wyaw ∈ [-0.8, 0.8]

**Sim2Sim 完整参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | Path | **必填** | 策略检查点 (.pt) 路径 |
| `--model` | Path | T1 XML | MuJoCo 模型文件路径 |
| `--duration` | float | 60.0 | 仿真时长（秒） |
| `--decimation` | int | 4 | 控制步间隔（匹配训练设定） |
| `--vx` | float | 0.3 | 初始前进速度 (m/s) |
| `--vy` | float | 0.0 | 初始侧移速度 (m/s) |
| `--wyaw` | float | 0.0 | 初始偏航角速度 (rad/s) |
| `--keyboard` | flag | 关闭 | **启用键盘控制**（启动时速度归零，按键调节） |
| `--headless` | flag | 关闭 | 无头模式（不打开可视化窗口） |
| `--no-realtime` | flag | 关闭 | 取消实时同步（全速运行） |

> **注意：** `--keyboard` 启用后，`--vx/--vy/--wyaw` 的初始值会被忽略（从零开始），需要通过按键操控。

---

## 3. 奖励函数设计（Reward Design）

本项目采用 **任务奖励 + 正则化惩罚 + AMP 风格奖励** 三层奖励结构。

### 3.1 任务奖励（Task Rewards）

| 奖励项 | 权重 | 公式 | 作用 |
|--------|------|------|------|
| `track_lin_vel_xy` | +1.25 | $\exp\left(-\frac{\|\mathbf{v}_{xy} - \mathbf{v}_{xy}^{cmd}\|^2}{0.25}\right)$ | 跟踪 XY 线速度指令 |
| `track_ang_vel_z` | +1.25 | $\exp\left(-\frac{(\omega_z - \omega_z^{cmd})^2}{0.25}\right)$ | 跟踪偏航角速度指令 |
| `is_alive` | +0.15 | 每步常数奖励 | 鼓励存活 |

### 3.2 基础惩罚（Base Penalties）

| 惩罚项 | 权重 | 公式 | 作用 |
|--------|------|------|------|
| `ang_vel_xy_l2` | -0.1 | $\|\omega_{xy}\|^2$ | 抑制横滚/俯仰角速度 |
| `flat_orientation_l2` | -1.0 | $\|g_{xy}^{proj}\|^2$ | 保持躯干水平 |

### 3.3 关节惩罚（Joint Penalties）

| 惩罚项 | 权重 | 公式 | 作用 |
|--------|------|------|------|
| `joint_vel_l2` | -2×10⁻⁴ | $\sum \dot{q}_i^2$ | 限制关节速度 |
| `joint_acc_l2` | -2.5×10⁻⁷ | $\sum \ddot{q}_i^2$ | 限制关节加速度，运动平滑 |
| `action_rate_l2` | -0.01 | $\sum (a_t - a_{t-1})^2$ | 动作平滑性 |
| `joint_pos_limits` | -1.0 | 超限位距离 | 关节限位保护 |
| `joint_energy` | -1×10⁻⁴ | $\sum |\tau_i \dot{q}_i|$ | 能量效率 |
| `joint_torques_l2` | -1×10⁻⁵ | $\sum \tau_i^2$ | 限制力矩 |

### 3.4 足部与接触惩罚

| 惩罚项 | 权重 | 公式 | 作用 |
|--------|------|------|------|
| `feet_slide` | -0.1 | 接触时脚的滑动速度 | 防止脚底打滑 |
| `soft_landing` | -5×10⁻⁵ | 着地冲击力 | 鼓励柔和着地 |
| `undesired_contacts` | -10.0 | 非足部刚体接触地面 | 防止摔倒 |

### 3.5 AMP 判别器奖励（Adversarial Motion Prior）

AMP 判别器作为额外的风格奖励信号，驱使策略生成接近人类动捕的自然运动：

| 参数 | 值 | 说明 |
|------|-----|------|
| 判别器架构 | MLP (1024, 512, 256) + ReLU | 三层隐藏层 |
| `amp_reward_coef` | 0.3 | AMP 奖励系数 |
| `task_style_lerp` | 0.7 | 任务-风格权重插值：$r = 0.7 \cdot r_{task} + 0.3 \cdot r_{amp}$ |
| 损失类型 | LSGAN | Least Squares GAN |
| `lsgan_reward_scale` | 0.25 | GAN 奖励缩放 |
| `grad_penalty_scale` | 10.0 | 梯度惩罚（正则化判别器） |

**判别器观测空间：** 基础角速度、关节位置、关节速度（连续 3 帧）  
**参考运动数据：** 8 组来自 ACCAD 数据集的女性行走动捕（前进、后退、启停、转弯）

### 3.6 对称性增强（Symmetry Augmentation）

利用人形机器人左右对称性，同时作为数据增强和镜像损失：

| 参数 | 值 | 说明 |
|------|-----|------|
| 数据增强 | 开启 | 将每个样本镜像翻转后加入训练批 |
| 镜像损失 | 开启 | $L_{mirror} = 0.2 \cdot \|a(s) - \text{flip}(a(\text{flip}(s)))\|$ |
| 镜像损失系数 | 0.2 | 约束左右对称动作 |

### 3.7 奖励设计思路

1. **多目标平衡**：任务奖励权重 (1.25) 远大于单项正则化惩罚 (10⁻⁴~10⁻²)，确保策略首先学会跟踪速度指令，再逐步优化运动质量。
2. **AMP 提供自然性**：纯 RL 奖励容易导致不自然的步态（滑步、僵硬等），AMP 判别器从人类动捕中学到的「自然运动分布」作为隐式奖励塑形，显著提升步态自然度。
3. **对称性约束**：通过数据增强和镜像损失，减少左右偏斜现象，加速收敛。
4. **能量/力矩惩罚**：兼顾 sim-to-real 迁移需要——过高的关节力矩在真实机器人上难以实现且有安全风险。
5. **接触惩罚**：`undesired_contacts` 权重为 -10.0（全场最大），强力阻止机器人用膝盖、手臂等非足部接触地面。

### 3.8 终止条件

| 条件 | 说明 |
|------|------|
| `time_out` | 回合超过 20 秒 |
| `fell_over` | 躯干倾角超过 70° |
| `illegal_contact` | 非足部刚体接触力超过 10N |

---

## 4. Sim2Sim 关键技术细节

Sim2Sim 验证是将 mjlab GPU 仿真训练的策略迁移到纯 MuJoCo CPU 仿真运行。以下三个关键点在开发中被发现和解决：

### 4.1 执行器模型匹配

mjlab 训练使用 **DcMotorActuator**（力矩直传 motor actuator + Python 端手动 PD 计算 + DC 电机饱和曲线），而非 MuJoCo 内建的 position servo。sim2sim 必须精确复现：

```
torque = kp * (q_target - q_actual) - kd * dq_actual
torque = dc_motor_clip(torque, dq_actual)  # 力矩-速度饱和曲线
data.ctrl[i] = torque  # 直接写入力矩（motor actuator, gain=1）
```

### 4.2 物理参数匹配

mjlab 使用 `cone=pyramidal, solver=newton`，而非 MuJoCo 默认的 `cone=elliptic, solver=CG`：

```python
model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
model.opt.iterations = 10
model.opt.ls_iterations = 20
```

### 4.3 关节顺序映射

策略的观测和动作均使用 **XML 运动树顺序**（kinematic tree order），而非执行器创建顺序。23 个关节中有 16 个在两种顺序下位置不同，需要构建排列映射。

---

## 5. 参考资料清单

### 论文

| 编号 | 论文 | 相关性 |
|------|------|--------|
| 1 | Peng et al., **"AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"**, ACM SIGGRAPH 2021 | 核心方法——对抗运动先验 |
| 2 | Escontrela et al., **"AMP for Real: Accelerated Physics-Based Character Animation"**, CoRL 2022 | AMP 在真实机器人上的应用 |
| 3 | Schulman et al., **"Proximal Policy Optimization Algorithms"**, arXiv 2017 | 基础 RL 算法 (PPO) |
| 4 | Todorov et al., **"MuJoCo: A physics engine for model-based control"**, IROS 2012 | 物理仿真引擎 |
| 5 | Loper et al., **"SMPL: A Skinned Multi-Person Linear Model"**, ACM SIGGRAPH Asia 2015 | 人体参数化模型 |
| 6 | AMASS: Archive of Motion Capture as Surface Shapes, ICCV 2019 | 动捕数据集（ACCAD） |
| 7 | Rudin et al., **"Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning"**, CoRL 2022 | 大规模并行 RL 训练范式 |
| 8 | Yu et al., **"Symmetric Data Augmentation and Mirror Loss for RL Locomotion"** | 对称性增强技术 |
| 9 | Ze et al., **"TWIST: Teacher-Student World Model for Efficient Sim-to-Real Transfer of Contact-Rich Manipulation"**, 2025 | GMR 全身遥操作参考 |

### 开源代码库

| 编号 | 项目 | 链接 | 用途 |
|------|------|------|------|
| 1 | **mjlab** | 内部框架 | MJWarp GPU 仿真 + AMP 训练 |
| 2 | **rsl-rl-lib** | https://github.com/leggedrobotics/rsl_rl | RSL 强化学习库 |
| 3 | **GMR** (General Motion Retargeting) | 内部工具 | 人体动捕数据重定向 |
| 4 | **MuJoCo** | https://github.com/google-deepmind/mujoco | 物理引擎 |
| 5 | **mink** | https://github.com/kevinzakka/mink | IK 求解器（GMR 使用） |
| 6 | **ACCAD 数据集** | https://accad.osu.edu/research/motion-lab | 动作捕捉数据 |

### 机器人平台

| 项目 | 说明 |
|------|------|
| **Booster T1** | 23-DOF 全身人形机器人 |
| MJCF 模型 | `src/mjlab/asset_zoo/robots/booster_t1/xmls/booster_t1.xml` (65 网格) |
| 自由度分布 | 腰部 3 + 左腿 6 + 右腿 6 + 左臂 4 + 右臂 4 = 23 DOF |

---

## 6. 常见问题

**Q: 训练前几千步机器人不动，正常吗？**  
A: 正常。AMP 训练初期判别器尚未收敛，风格奖励信号弱，策略倾向于站立以获取 `is_alive` 奖励。约 5000 迭代后开始出现行走行为。

**Q: Sim2sim 中机器人摔倒？**  
A: 检查三个关键匹配项：(1) 执行器模型是否为 motor + 手动 PD；(2) 接触锥是否为 pyramidal + newton 求解器；(3) 关节顺序映射是否正确。

**Q: 键盘控制无响应？**  
A: 键盘输入通过终端 (termios) 读取，确保在运行脚本的终端窗口中按键，而非在 MuJoCo 查看器窗口中。

---

## 许可证

本项目仅供学术研究使用。机器人模型版权归 Booster Robotics 所有。
