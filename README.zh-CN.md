# OneForward

**直接读取 Qwen3.5-2B logits 的零训练、零解码类型化决策服务。**

[English](README.md) · [API 示例](#api) · [实现原理](#实现原理) · [能力边界](#能力边界)

OneForward 将文本、图片和视频交给未经修改的开源权重
[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 包装成一个小型的
Jev 风格决策服务。它动态构造 prompt，把用户定义的选项映射到单 token
标签 `A`–`H`，每个 Choice 问题只执行一次模型 forward，随后直接摘取下一
token 的 logits。整个过程没有生成式解码、采样、微调、adapter 或任务专用权重。

> [!IMPORTANT]
> OneForward 是受 Jev/System One 启发的独立研究项目，与 TypeSafe AI
> 没有关联，也未获得其背书。本项目不复现 Jev 的专有训练、概率校准、服务系统、
> 速度或语义能力；目前只兼容公开接口中一个实用的 Choice 请求/响应子集。

![OneForward 真实 Qwen3.5-2B logits 决策界面](docs/assets/playground.png)

## 核心特点

- **无需额外训练**：原样使用上游 Qwen3.5-2B 权重。
- **不生成答案**：每个问题一次 forward，返回 `output_tokens: 0`。
- **原生多模态证据**：每次请求最多加入 8 张图片或 1 个视频，视觉编码器与
  决策 prompt 共同参与同一次 forward。
- **选项运行时定义**：请求中直接给出选项名称与语义，无固定分类头。
- **直接返回概率**：对允许的标签 token logits 做 softmax，无需解析生成文本。
- **过程透明**：`_debug` 提供候选 logits、token ID、candidate mass、完整 prompt
  和全词表 top tokens。
- **完全自托管**：核心依赖只有 FastAPI、Transformers 与开源权重。

## 实现原理

假设选项为 `billing`、`technical` 和 `sales`，prompt 中会出现：

```text
A. billing: Payment or subscription issues
B. technical: Bugs or integration problems
C. sales: Pricing or account questions

Return exactly one option letter from A, B, C.
```

程序使用 Qwen 官方 chat template 并关闭 thinking。设最后位置的全词表 logit
向量为 `z`，选项标签的 token ID 为 `t_i`：

```text
p(option_i | allowed options, prompt) = softmax(z[t_i] / temperature)
choice = argmax_i z[t_i]
```

程序还会返回 `candidate_mass`，即全部允许标签在全词表下一 token 分布中的
总概率质量。它可以识别一种常见假象：A/B/C 内部归一化后看起来很确定，但模型
本来根本不想输出 A、B 或 C。

服务启动时会验证每个答案标签都恰好是 prompt 后的一个 tokenizer token；如果
tokenizer 或 chat template 更新破坏该条件，程序会直接报错，而不是静默地产生错误结果。

## 快速开始

需要 Python 3.10+。Qwen3.5-2B 权重约 4.6 GB；推荐使用 NVIDIA GPU。代码
包含 CPU fallback，但尚未做性能测试。

```bash
git clone https://github.com/Embodied-AI-System/Qwen3.5-OneForward.git
cd Qwen3.5-OneForward

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
./run.sh
```

首次启动会从 Hugging Face 下载 `Qwen/Qwen3.5-2B`。模型就绪后打开
<http://127.0.0.1:8000>。

已有本地权重时：

```bash
JEV_MODEL_PATH=/absolute/path/to/Qwen3.5-2B ./run.sh
```

## API

顶层 `media` 字段可选，所以原有纯文本客户端保持兼容。多模态请求用 base64
data URL 传入图片或视频：

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Customer says the integration keeps failing. Please help ASAP.",
    "model": "jev-latest",
    "media": [{
      "type": "image",
      "name": "scene.png",
      "data_url": "data:image/png;base64,..."
    }],
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      }
    }
  }'
```

一次真实 BF16 运行返回 `technical`，对应 Choice 内部概率为 `0.997197`，
`candidate_mass` 为 `0.999844`，且 `output_tokens` 为 `0`。完整返回示例见
[英文 README](README.md#api)。

网页支持选择或拖拽文件并直接预览。API 支持 PNG、JPEG、WebP、GIF、MP4、
WebM、MOV、MKV 和 AVI。单张图片上限 16 MiB；视频上限 64 MiB、180 秒，
默认按 1 FPS 且最多 32 帧采样。服务只在本地解码，返回值不会回显 base64，
`_debug.media` 只包含尺寸、字节数、时长和采样帧数。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `JEV_MODEL_PATH` | `Qwen/Qwen3.5-2B` | Hugging Face 模型 ID 或本地目录 |
| `JEV_DEVICE` | 有 CUDA 时为 `cuda`，否则 `cpu` | PyTorch 设备 |
| `JEV_PROMPT_MODE` | `chat` | `chat` 使用官方模板；`raw` 使用 `Answer:` 补全模板 |
| `JEV_TEMPERATURE` | `1.0` | 只缩放候选 logits，不会触发采样 |
| `JEV_MAX_INPUT_TOKENS` | `8192` | 文本与视觉 token 的总上限 |
| `JEV_VIDEO_FPS` | `1.0` | 视频目标采样率 |
| `JEV_MAX_VIDEO_FRAMES` | `32` | 每个视频最多采样帧数 |
| `JEV_HOST` | `127.0.0.1` | 监听地址 |
| `JEV_PORT` | `8000` | HTTP 端口 |
| `JEV_PYTHON` | `python3` | `run.sh` 使用的 Python |

## 测试

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m compileall -q app.py core.py inference.py media.py tests scripts
```

服务启动后可以执行真实模型 smoke test：

```bash
./scripts/smoke_test.py
```

## “零训练”的准确含义

本仓库不会训练或修改任何模型权重，只在官方 `Qwen/Qwen3.5-2B` checkpoint
外增加 prompt 构造、单 token 候选验证、logit 摘取、概率归一化、HTTP API 与网页界面。

上游 Qwen3.5-2B 本身是 Qwen 官方发布的 **post-trained** 模型。这里的准确
表述是：OneForward 没有做任何*额外的*微调、后训练、强化学习、蒸馏或校准。

## 能力边界

- 当前仅支持 2–8 个选项的 Choice；不支持 Noul、Score 或更大候选集。
- 多模态输入需要默认的 chat prompt 模式；raw `Answer:` 基线仍只支持文本。
- 每次请求最多 8 个媒体附件且最多 1 个视频；视频会抽帧，音频不会输入模型。
- 多问题按顺序执行，暂未实现 batch 或共享 prefix cache。
- 选项概率是允许标签集合内的条件概率，不是“答案正确”的校准概率。
- `confidence = 1 - normalized_entropy(probabilities)` 只是实验性启发式公式，
  不是 Jev 的 confidence 实现。
- 选项顺序与标签 token 可能引入偏差；正式使用前必须在自己的标注数据上测试排列扰动。
- 项目示例只是连通性检查，不构成准确率或校准 benchmark。
- 当前没有鉴权、限流或生产加固。

在没有领域评测和人工保护措施的情况下，请勿将本研究预览用于高风险或不可逆决策。

## 许可证与署名

本仓库代码采用 [Apache License 2.0](LICENSE)。仓库不包含 Qwen3.5-2B
权重；权重仍受其[原始许可证](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/LICENSE)
约束。

Jev 与 System One 仅用于说明启发本项目的接口模式，相关名称和商标归各自权利人所有。
