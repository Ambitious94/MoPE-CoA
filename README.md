# MoPE-Transformer (Mixture of Pipeline Experts)

MoPE-Transformer把 Transformer 的 FFN 替换为“管线专家的混合体（MoPE）”。每个“专家”是一条可执行的推理/工具链（如：plan → retrieve → read → verify）。本项目已经从“最小可用原型”升级为可工程化使用的组件集，支持任务型搜索/QA/事实核查，并提供命令行、HTTP 服务与 nanoGPT 集成适配器。

## Motivation
Traditional FFNs apply a uniform nonlinear transformation to every token. MoPE treats the FFN slot as a **structural decision point** where the model can pick among multiple executable pipelines with different cost/accuracy trade-offs. This enables:

- **Adaptive strategy selection:** choose cheap or strong pipelines per token or step.
- **Explicit reasoning traces:** every pipeline emits a step-by-step trace for auditability.
- **Task-aware computation:** pipelines integrate search, retrieval, comparison, and verification without leaving the Transformer graph conceptually.

## Repository layout
- `mope/` — 包含门控、管线、向量化、检索/阅读、任务引擎、nanoGPT 集成、CLI 与服务端。
- `docs/` — 架构与训练思路。
- `tests/` — Pytest 覆盖路由/检索/任务引擎与 MoPE 层的端到端行为。

## Quick start
在 Windows PowerShell 下：

1) 安装依赖（包含 pytest 便于本地测试）

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) 运行测试（验证基础功能端到端）

```powershell
pytest -q
```

3) 使用命令行进行索引与问答

```powershell
# 将当前目录下的 txt 文本索引到快照文件（默认为 ./.mope_store.json）
python -m mope.cli index --dir .

# 提问：检索 → 阅读 → 答案综合 + 轻量事实核查
python -m mope.cli ask "When does water boil?" --top-k 3 --trace
```

4) 启动 HTTP 服务（可选）

```powershell
pip install "fastapi>=0.110" "uvicorn>=0.24"
python -c "from mope.server import build_app; import uvicorn; uvicorn.run(build_app(), host='0.0.0.0', port=8000)"
```

请求示例：

```powershell
curl -X GET http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/index -H "Content-Type: application/json" -d "[{\"id\":\"doc1\",\"text\":\"Water boils at 100 C at sea level.\"}]"
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"question\":\"When does water boil?\",\"top_k\":3}"
```

## Key ideas
- **Pipelines as experts:** a pipeline is an ordered chain of atomic reasoning tools (planner, search, reader, verifier). Different chains capture different search/QA strategies.
- **Gate-driven routing:** a gate projects the hidden state to select the most promising pipeline while keeping probability distributions for analysis.
- **Vectorization:** pipeline textual outputs are mapped back to vector space so the Transformer can continue computation.
- **Training curriculum:** start with FFN distillation for stability, then supervised fine-tuning on search data, and finally RL to trade off accuracy, cost, and factuality.

详见 `docs/architecture.md`。

## Features（工程化能力）
- 任务引擎：`SearchQAFactCheckingSystem` 支持检索、阅读、答案综合与轻量事实核查。
- 可插拔 MoPE 层：`MoPELayer` 以门控选择不同的“策略管线”，并将文本输出映射回向量空间继续计算。
- nanoGPT 集成：`attach_mope_to_nanogpt` 将指定层的 MLP 替换为 MoPE 适配器（支持 list 与 torch.Tensor）。
- CLI：
   - `index` 从目录（.txt）或 JSONL 索引文档到快照文件
   - `ask` 进行检索→阅读→回答/核查，支持 trace 输出
- HTTP 服务：基于 FastAPI 暴露 `/index`, `/ask`, `/health` 端点，便于系统集成。
- 跨平台 CI：GitHub Actions 在 Windows/Ubuntu、Python 3.9–3.11 上运行测试。

## API 速览
- 检索与阅读：`mope.retrieval.DocumentStore`, `EvidenceReader`, `build_retrieval_pipelines`
- 任务型搜索/QA/核查：`mope.task_engine.SearchQAFactCheckingSystem`, `build_task_pipelines`
- MoPE 层与模型：`mope.mope_layer.MoPELayer`, `mope.model.MoPETransformer`
- nanoGPT 适配：`mope.nanogpt_integration.attach_mope_to_nanogpt`, `make_mock_nanogpt`

## HF/Qwen 集成与训练阶段建议

推荐训练顺序（稳定性优先）：

1) 阶段 1：结构对齐（蒸馏）
      - teacher：原始 Qwen（不插 MoPE）
      - student：插入 MoPE 的 Qwen（仅替换末端 1–2 层），`alpha≈0.01`
      - loss：KL(logits) 为主，MSE(hidden) 可选
      - 目标：插入 MoPE 后，模型仍能“像原来一样说话”

2) 阶段 2：SFT（CoA/QA）
      - 输出强制 `<think>...</think><answer>...</answer>`
      - 观察 `observation` 字段不计入 loss（mask）
      - 先用离线工具回放，稳定格式后再接真实检索/爬取

3) 阶段 3：Pipeline Experts（文本→向量）
      - pipeline 的文本输出需经轻量 encoder/pooling 映射回 `hidden_size`
      - MoPE 始终以 `hidden = hidden + alpha * delta` 的残差形式输出
      - 分阶段引入，避免一次性耦合过多改动

Quick start（冒烟/蒸馏）：

```powershell
# 冒烟：加载 Qwen、在指定层挂 MoPE 并生成（36 层：末端两层为 34,35）
python -m scripts.qwen.smoke `
      --model Qwen/Qwen2.5-3B-Instruct `
      --layers 34,35 `
      --alpha 0.01 `
      --dtype float16 `
      --device-map auto

# 蒸馏：teacher→student（仅训 gate/alpha；如需也训练适配器，加入 --train-adapters）
python -m scripts.qwen.distill `
      --teacher Qwen/Qwen2.5-3B-Instruct `
      --student Qwen/Qwen2.5-3B-Instruct `
      --layers 34,35 `
      --alpha 0.01 `
      --max-steps 200 `
      --lr 1e-4 `
      --dtype float16 `
      --device-map auto
```

gate.json 兼容性：
- `gate.weight` 形状必须为 `[hidden_size, num_experts]`；`gate.bias` 为 `[num_experts]`
- 从 GPT-2（768）迁移到 Qwen（更大维度）时，旧 gate.json 不可直接复用；建议重新初始化并短训校准

## nanoGPT 集成示例（Legacy）
### nanoGPT SFT（分目录）

在新的分目录结构下，使用包装脚本运行统一的 SFT：

```powershell
# 查看参数帮助（与旧脚本一致）
python -m scripts.nanogpt.sft --help

# 示例训练（请根据你的路径与数据修改）
python -m scripts.nanogpt.sft `
      --input "data/WebAgentSFTDataset.json" `
      --nanogpt-root "e:/Edge Download/nanoGPT-master" `
      --model-type gpt2 `
      --layer-idx 0 `
      --unfreeze-last 1 `
      --epochs 3 `
      --batch-size 8 `
      --grad-accum-steps 4 `
      --lr 5e-5 `
      --weight-decay 0.01 `
      --warmup-steps 100 `
      --amp `
      --out gate.json
```

```python
from mope.retrieval import DocumentStore
from mope.task_engine import SearchQAFactCheckingSystem, build_task_pipelines
from mope.nanogpt_integration import attach_mope_to_nanogpt, make_mock_nanogpt

store = DocumentStore()
store.add("doc1", "Water boils at 100 degrees Celsius at sea level.")
system = SearchQAFactCheckingSystem(store)
pipelines = build_task_pipelines(system)

mock_model, hidden_size = make_mock_nanogpt(num_layers=1, hidden_size=8)
attach_mope_to_nanogpt(
      mock_model,
      hidden_size=hidden_size,
      layer_indices=[0],
      pipelines=pipelines,
      prompt_provider=lambda: "When does water boil?",
)

updated = mock_model.transformer.h[0].mlp([0.0]*hidden_size)
```

## 设计取舍与局限
- 检索为轻量化 BM25 风格（纯 Python），适合演示与研究；生产可替换为外部检索（ES/FAISS 等）。
- 向量化使用哈希 BOW，仅用于把文本回写到向量空间，不与底层 LM 权重对齐。
- nanoGPT 适配器仅更新最后一个时间步；张量原位写回需安装 PyTorch。

## 贡献与开发
- 安装开发工具（可选）：

```powershell
pip install ruff mypy
```

- 代码质量检查：

```powershell
ruff check . ; mypy mope
```

---

如需将检索替换为向量库或外部搜索，或扩展更复杂的事实核查器，欢迎提 Issue/PR。
