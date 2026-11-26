# sciagent — 证据可追溯的科研数据甄别 Agent

一个**通用、可复用**的 Python 框架：用 API key 调用 LLM，让它对科研数据进行
"甄别"（识别、质控、结论支撑度判断），但 **LLM 不直接下结论**——它调用
Python 数值分析工具、检索本地知识库，拿到结构化证据后，再给出
**结论 + 引用 + 置信度 + 必要时拒识**。

当前内置一个完整示例实例：**拉曼光谱化学物质识别与证据分析**（基于
Flanagan & Glavin, Scientific Data 12:498, 2025 的公开数据集）。

---

## 1. 为什么这样设计（核心思想）

普通做法容易让 LLM"看图猜谱峰"或"编造文献"。本框架的约束是：

```
用户问题
   │
   ▼
LLM 决策要调用哪个工具（function calling）
   │
   ▼
Python 工具返回【结构化证据】  ←── 真正的数值分析在这里完成
   │
   ▼
LLM 基于证据生成结论，必须引用工具结果；证据不足 → 拒识
```

- 数值分析（相似度、找峰、异常检测）由 Python 工具完成，**不塞进 prompt**。
- 知识库检索提供参考信息（分子式、参考谱峰、供应商）。
- 系统提示词强制"只引用工具结果、不编造、不确定就拒识"。

## 2. 目录结构

```
llm_sci_agent/
├── main.py                    # 命令行入口
├── webui.py                   # Web 交互界面（Streamlit：Key→导入→识别→导出）
├── run.bat / run_webui.bat / setup.bat  # 一键运行 / 打开 Web / 初始化环境
├── models/                    # 本地语义模型（离线，可选）
├── sciagent/                  # 核心包
│   ├── config.py              # 读 .env / 环境变量
│   ├── llm.py                 # OpenAI 兼容客户端
│   ├── tools.py               # 工具注册表
│   ├── agent.py               # Agent 编排循环 + 系统提示词
│   ├── knowledge.py           # 极简本地知识库
│   ├── retrieval.py           # 向量检索（TF-IDF，中英文，可换语义模型）
│   ├── preprocess.py          # 光谱预处理（SG平滑 + ARPLS基线 + SNV）
│   ├── tools_data.py          # 通用数据质控工具（CSV/XLSX）
│   ├── tools_raman.py         # 拉曼工具（识别/找峰/检索/开集拒识）
│   └── app.py                 # 组装：LLM+工具+知识库
├── scripts/
│   ├── generate_assets.py     # 从源数据生成知识库/质心/示例谱
│   ├── calibrate_open_set.py  # 开集拒识校准（类内 z 分数阈值）
│   ├── eval_groundedness.py   # groundedness 自动评估
│   └── eval_robustness.py     # 分布偏移/跨源鲁棒性评估
├── knowledge/                 # 生成的知识库（reference.json + centroids.npz + 校准）
├── reports/                   # groundedness_report.json 等评估结果
├── examples/
│   ├── raman_demo.py          # 拉曼识别示例
│   └── query_spectra.npz      # 示例查询光谱
├── requirements.txt           # 核心依赖
├── requirements-semantic.txt  # 语义检索可选依赖（sentence-transformers）
└── .env.example / .gitignore / README.md
```

## 3. 快速开始

### 3.0 一键运行（绿色免安装，新电脑解压即可用）

项目**完全自包含**，解压到任意电脑（Windows）后无需联网、无需装 Python：

- `runtime/`：内置完整 Python 3.13.18 + 全部依赖（requests / numpy / pandas / scipy /
  scikit-learn / matplotlib / openpyxl + torch CPU + sentence-transformers + streamlit + plotly），
  绿色可移植，**不再依赖 .venv 或系统 Python**。
- `models/BAAI/bge-small-zh-v1.5/`：语义检索模型已本地化，`config.py` 会自动优先
  加载本地模型，语义后端同样离线运行。

直接双击 `run.bat`（或在命令行 `run.bat --example 0`、`run.bat --example 6`）即可，
不用 `pip install`、不用先跑 `setup.bat`。

**Web 界面**：双击 `run_webui.bat` 会在浏览器打开交互界面，三步完成——
① 侧边栏填 API Key（**必填**，匹配由 Agent 完成）→ ② 上传光谱 CSV 或选内置示例 →
③ 一键识别：**Agent（大模型 function-calling）亲自调用匹配工具**，返回 Agent 的分析结果，
并附置信度、候选表、谱图，可下载结果（JSON / CSV / 光谱图 HTML）。
上传的 CSV 若只是完整谱的一段（部分区间，如指纹区 800~1800 cm⁻¹），系统会按波数
自动定位它落在参考库的哪个区间，Agent 调部分区间匹配工具做识别。
**如果只给了少量峰值数据（几个到几十个离散峰位+强度，没有连续谱）**，系统会自动切到
「峰位匹配」模式：直接拿这些峰位和参考库每条化合物的权威谱峰（`raman_bands_cm1`，
从源元数据提取）做加权比对（峰强权重 × 区分度 IDF × 距离衰减），不做连续谱插值，
因此即使只给 3~15 个特征峰也能稳定命中正确化合物（已用 6 个真实示例在 N=3/5/8/10/15
下验证 Top-1 全命中）。识别为**全自动**：

- 本地匹配置信度为「高」→ 输出 Agent 的本地匹配结论（候选表 + 谱图 + 导出）；
- 置信度达不到「高」→ **不输出本地匹配结论**，自动让大模型结合实验峰位联网检索，
  多源查证（PubChem / NIST WebBook / SDBS），用 Bing 等做峰位佐证搜索，给出可能的
  物质类型与候选（SDBS 国内直连常超时，结果里会给出 SDBS 检索链接）。

> 唯一需要联网的是：识别时的 API Key（调用 DeepSeek 联网分析）。本地库/模型/依赖
> 全部离线。`setup.bat` 仅作为 `runtime/` 万一损坏时的手动重建入口，一般无需运行。

### 3.1 安装依赖（手动/新机器）

```bash
pip install -r requirements.txt
```

> 核心链路（LLM 调用 + 编排）其实只依赖 `requests`；`numpy/pandas/scipy/…`
> 是数据工具需要的。

**可选：语义检索后端**（默认 TF-IDF 不需要）。要用中文语义检索再装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU 版，体积小
pip install -r requirements-semantic.txt
```

### 3.2 配置 API key

```bash
copy .env.example .env      # Windows
# 编辑 .env，填入：
#   LLM_API_KEY=你的key
#   LLM_BASE_URL=https://api.deepseek.com
#   LLM_MODEL=deepseek-flash      # 或 deepseek-v4-pro
```

**key 只放 `.env`，不进代码、不进 git**（`.gitignore` 已忽略）。

检索后端通过 `.env` 切换（默认 `tfidf`，无需模型）：

```bash
# .env 追加（要用语义检索时）
EMBEDDER_BACKEND=semantic
EMBEDDER_MODEL=BAAI/bge-small-zh-v1.5
HF_ENDPOINT=https://hf-mirror.com   # 国内下载模型走镜像
```

### 3.3 生成知识库（首次需要，依赖源数据）

```bash
python scripts/generate_assets.py
```

会生成 `knowledge/raman_reference.json`、`knowledge/raman_centroids.npz`、
`examples/query_spectra.npz`。**只读源数据，不修改。**

### 3.4 跑一次识别

```bash
python main.py "请识别这条拉曼光谱属于哪种已知物质，给出候选、证据与置信度"
python main.py --example 1      # 换第 1 条示例光谱
python main.py --example 6      # 合成噪声，观察它是否会"拒识"
```

或看带注释的示例脚本：

```bash
python examples/raman_demo.py
```

### 3.5 可选：开集校准与可靠性评估

```bash
# 开集拒识校准：用整类留出法评估，生成 knowledge/open_set_calibration.json
python scripts/calibrate_open_set.py

# groundedness 评估：跑一组预设问题，统计工具调用率/事实正确率/引用率
python scripts/eval_groundedness.py

# 分布偏移/跨源鲁棒性：模拟不同采集条件（噪声/荧光/校准偏移/降分辨率）
python scripts/eval_robustness.py
```

`raman_identify` 会自动读取校准文件，用「类内 z 分数」判断库外/未知（不是只看原始余弦相似度）。

## 4. 怎么换成你自己的科研数据

把"拉曼"替换成你的领域，只需两步：

1. **写工具**：实现返回 dict 的 Python 函数（如 `my_analysis(x)`），用
   `tool("名称","描述", JSON Schema)(函数)` 注册进 `ToolRegistry`。
2. **换知识库**：往 `KnowledgeBase` 里放你的参考数据（JSON 即可）；`VectorRetriever`
   会自动为这些条目建向量索引，并注册 `search_knowledge` 检索工具（可在条目里加
   `keywords` 字段提升检索效果）。

核心的 `config / llm / tools / agent` 完全不用改。

## 5. 安全与诚实性说明

- API key 永远从 `.env` / 环境变量读取。
- 系统提示词强制"证据可追溯、证据不足就拒识"，但 LLM 仍可能犯错：
  **数值匹配 ≠ 化学等同**，生产使用前请对工具返回做二次校验。
- 知识库里已知的元数据错误（如四氢呋喃参考谱峰被误复制）已用 `note` 字段标注，
  提醒 Agent 不要据此下结论。

## 6. 已知限制

- 知识库检索支持两种后端：TF-IDF（默认，离线免下载、含中文 n-gram）与
  sentence-transformers 语义向量（`EMBEDDER_BACKEND=semantic`，默认模型
  `BAAI/bge-small-zh-v1.5`）。语义模型对化学领域极冷门术语（如"羰基"）仍可能
  召回不准，建议结合 `keywords` 字段，或让 Agent 用 `raman_list_compounds` 交叉核对。
- 示例只用了公开纯物质数据集，且每类约 100 条谱来自同一物理样品的重复扫描，
  因此"跨瓶/跨仪器/跨来源泛化"尚未验证。
- 开集拒识已用「类内 z 分数」校准（衡量对最近库类的典型性），在本数据集留出类上
  AUC≈0.9999；但对"与库内化合物光谱几乎重合的近重复物"，任何纯谱匹配都无法绝对区分，
  需正交手段（GC-MS/NMR）交叉验证。
- 上位概念查询（"羰基/羟基/含氧/卤素/非极性/苯环"等）通过内置同义词扩展
  `expand_query` 召回库内子类；更冷门术语仍建议配合知识库 `keywords` 字段。
- 跨源泛化用受控扰动模拟（加性噪声/荧光基线/校准偏移/降分辨率），实测 Top-1 波动
  约 1% 以内；真实第二来源数据可通过 `eval_robustness.py` 里的 `load_external()` 接入。
