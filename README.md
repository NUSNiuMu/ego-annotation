# ego-annotation

从现有自动化提取的独立标注流程：

```text
本地 MCAP → 按原始时间戳采样 JPEG → Qwen 窗口候选
                                      ↓
                     整条文本审理与 episode 分割
                                      ↓
                         仅具体疑点做图像复核
                                      ↓
                 接纳标注 + 待复核内容 + 排除项 + 证据
```

每张卡一个模型、最多两个录制的窗口混合推理，短录制完成即交给审核并补位；
最多两个独立审核任务，与 Qwen 重叠运行。不会拉 NAS、处理手部、导出数据包、
上传 OSS 或发布 Viewer，也不连接原生产服务。模型权重、客户数据和认证不在仓库中。

## 安装

Linux、Python 3.10+、NVIDIA GPU；Qwen3.5-9B 权重须自行准备在本地。
当前模型加载器固定 Qwen3.5，沿用 FP16、SDPA 和 JPEG 输入。
24GB TITAN RTX 是来源流程的环境，不代表已在其他硬件验收。

```bash
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -e '.[inference]'
npm install -g @openai/codex
codex login
codex login status
cp configs/annotation.example.json configs/local.json
```

无浏览器的主机可使用 `codex login --device-auth`，需账号或工作区允许设备登录。
请使用自己的账号独立授权；详见 [OpenAI 官方认证说明](https://developers.openai.com/codex/auth/)。
需要能访问审核服务；GPT 型号和调用权限以自己的账号为准，在配置中明确填写。
调用沿用来源流程的 Codex CLI 参数；不同 CLI 发行版须先核对 `codex exec --help`。

推理可选依赖保留来源流程的 Torch/Transformers 版本。高效 DeltaNet 的
`flash-linear-attention`、`causal-conv1d` 需结合 GPU/CUDA 编译环境自行安装；
首窗回执记录实际加速包和函数调用。仅安装基础依赖不能声称复现生产吞吐。

## 同事需要修改的部分

仓库提供的是**可运行的叠衣服示例**，尚未适配其他任务和相机。

| 位置 | 修改内容 |
| --- | --- |
| `configs/local.json` | 本机模型目录、GPU 编号、相机 topic、采样参数、GPT 型号和 Codex 路径 |
| `src/ego_annotation/sampling.py` | 当前只读 MCAP 的 CDR `sensor_msgs/msg/CompressedImage`、Annex-B H.264；其他格式或多视角对齐在此适配 |
| `src/ego_annotation/inference/schema.py` | `prompt()` 中的 Qwen 任务描述；保留输出 JSON 结构 |
| `skills/garment-review-policy.json` | 任务起止、完成标准、英文动作表、疑点规则 |
| `skills/ego-text-review/SKILL.md` | 该任务的文本整理与分段规则 |
| `skills/ego-image-review/SKILL.md` | 该任务具体疑点的图像判断规则 |
| `skills/review-contract.md`、`review.py`、`review_transport.py` | 若修改字段或疑点类型，必须同步协议、校验器和模型输出 schema |

历史 `garment-*` 字段是当前协议标识，不表示已经支持任意任务；
保留字段结构、只修改动作表及规则时可继续使用此协议。
`GARMENT_IDENTITY`、`PHONE_INTERRUPTION` 等类型与示例相关；其他语义需要显式改协议。
手机问题跳过图审的分支位于 `review_runner.py`；更换此规则时同步修改示例 policy。

当前只使用指定 topic 的真实单视角证据，不声称多相机审核。
有多路画面时同事须实现真实时间戳映射，再扩展采样包与图审附件。
MCAP 要有 summary，帧时间严格递增，一条消息对应一帧 H.264 access unit，
首帧包含 SPS/PPS/IDR；不满足时明确失败，不猜时间或静默换解码方案。

## 运行

先改 `configs/local.json`，在自己的机器用一条数据验证采样和新任务规则：

```bash
ego-annotate --config configs/local.json \
  --source /data/raw/take001 --output /data/annotations --prepare-only

ego-annotate --config configs/local.json \
  --batch-root /data/raw --output /data/annotations
```

`--source` 可重复传入 MCAP 文件或录制目录；`--batch-root` 按目录组织多个录制，
同目录的 MCAP 是同一录制的分片。修改代码或配置后不承诺跨版本缓存复用。
配置内的相对路径以配置文件所在目录为基准。输出必须独立于原始输入和模型目录。

同一输出目录只有一个控制器。GPU 启动前查询 UUID、拒绝占用或故障卡，
CUDA 加载后再校验 UUID；本仓库不配置原服务器的六卡映射或借还调度。
任务失败独立记录，其他录制继续；模型加载失败终止本次执行并留下 `failure.json`，
不自动重启模型。重新执行命令会复用同版本的完成窗口和审核结果，只补缺失阶段。
审核只对有回执证实的传输失败做有限重试，不因语义不满意重复调用。
已保存模型调用仍存活或证据损坏时停止并要求显式检查。

## 输出和验收

```text
<output>/
  batch.json                         每条录制的 Qwen 与审核状态
  sources/<source_id>/prepare-*/      时间戳、帧映射、采样图及采样回执
  jobs/<job_id>/
    job.json / status.json           输入、代码、配置和模型绑定
    windows/*.json                   Qwen 原文和逐窗校验
    segments.json / summary.json     候选动作与统计
    review/<method_id>/
      method/                        本次固定使用的两个 skills 和规则
      text-call-*/ / image-call-*/    原始调用、附件哈希、用量、耗时、返回码
      final/episodes.json            可接纳 episode 与动作标注
      final/deferred.json            待复核 episode、局部执行问题
      final/exclusions.json          排除项
      final/receipt.json             程序校验及审核依据
      complete.json / state.json     完成证据与当前状态
```

Qwen 坏窗口保存原文，强制局部图审；只隔离受影响的 episode。
无图像疑点时不调用图审。边界和完成明确、内部动作不明确时可留白。
不生成检查页面/视频，不自动删除输入或证据。

`COMPLETE` 表示计算和保存完成；是否有可用标注看 `accepted_episodes`，
无法判断的内容看 `deferred_episodes`。空 Qwen 结果不等于整条录制有合格标注。
这也不代表人工金标、手部处理或任何下游接纳。

```bash
python -m unittest discover -s tests
ego-annotate --help
```

提取验收见 [reports/validation.md](reports/validation.md)。单元测试使用合成 MCAP 和
明确标注的模型替身，不冒充新任务真实推理验收。
