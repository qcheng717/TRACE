# TRACE

TRACE 是面向多任务机器人操作的技能表征学习框架，通过语言条件视觉特征融合和未来特征变化预测学习共享技能字典，再通过路由器与扩散策略生成动作。本仓库提供 RoboTwin 上的数据预处理、两阶段训练和评测实现。

## 核心设置

- 单路 head RGB，320×240，无增广；冻结 DINOv2-base 四层特征与 MiniLM。
- 语言条件特征融合、空间 Transformer、15×100 共享技能字典。
- 阶段一预测清理后序列的 t+32 特征变化；10K 冻结 online/target selector，继续至 20K。
- 阶段二冻结预测器，训练路由器和 diffusion policy 至 100K。
- 输出活动臂 6 个绝对关节角和夹爪，chunk=32；下方英文 README 的评测命令使用 horizon=16。
- 逐维全局 min/max；动作尾部重复末动作；完整 chunk 反归一化后进行 B-spline 平滑。

## 使用

完整安装、数据下载和参数说明见 [英文 README](README.md) 与 [协议文档](docs/protocol.md)。安装 PyTorch 及独立 Python 3.10 训练环境后：

```bash
pip install -e '.[dev]'
bash scripts/setup_robotwin.sh
python tools/download_models.py
bash scripts/download_data.sh
bash scripts/prepare_data.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 TRACE_NPROC=7 TRACE_BATCH=128 \
TRACE_RUN=outputs/trace_robotwin bash scripts/train_two_stage.sh
```

运行模拟器评测前，需要按照 RoboTwin 官方说明安装模拟器环境和资源。训练和模拟器可使用不同 Python，通过 `TRACE_POLICY_PYTHON` 指定训练环境。

README 中的三段演示来自已有的真实成功 rollout，附有来源、seed 和 SHA256；它们是定性示例，不代表总体成功率。

发布范围：仅源码、配置、说明、测试和少量视频。MIT 用于本仓库代码；第三方模型、数据和资源仍遵循各自授权。
