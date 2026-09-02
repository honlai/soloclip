# soloclip

**solo** = 画面里只有一个人、声音里也只有一个人。

给一份影片清单，自动剪出**每部一支、20 秒左右、同一个人正对镜头说话、且没有其他人声**的片段。
画面不合格但声音乾淨时，退而输出同样长度的纯音讯。

```
清单  →  人脸辨识  →  声纹比对(选用)  →  切割拼凑  →  转档(选用)
```

实跑成绩：355 支网址 → 248 支成品（影片 217、纯音讯 31），
其中 127 支（51%）是完全没有接点的单一连续镜头。

English: [README.md](README.md)
繁體中文說明：[README.zh-tw.md](README.zh-tw.md)

---

## 五个步骤

### 1 · 给清单

一行一个网址，放进 `url_list*.txt`。设定档指定要读哪些：

```yaml
paths:
  url_list:
    - ../url_list.txt
    - ../url_list_2.txt      # 多份清单会依 video id 去重
```

新增清单不会重跑旧的：已经产出成品的影片会被跳过。

### 2 · 人脸辨识

对音讯乾淨的区间抽帧，逐帧判断四件事，全部通过才算「这个人正在对镜头说话」：

| 条件 | 判准 | 参数 |
| --- | --- | --- |
| 脸够大 | 脸宽佔画面比例 | `asd.min_face_ratio` |
| 正对镜头 | 头部 yaw / pitch 各自设限 | `asd.max_yaw_deg` / `max_pitch_deg` |
| 是同一个人 | 脸部 embedding 余弦距离 | `asd.face_id_threshold` |
| 真的在说话 | 对齐后嘴部区域的帧差能量 | `asd.lip_motion_min` |

身分基准不预设「最大的脸就是主角」，而是取主讲者说话期间所有脸的最大群集中心，
所以旁边有观众入镜也不会选错人。

**逐帧原始测量值会存成 `.npz`**，调门槛不必重跑 GPU：

```bash
soloclip -c configs/talks.yaml asd --rescore     # 套用新门槛，秒级
python tools/sweep.py --yaw 30 35 40 --size 0.06 0.05      # 扫描并回报实际成品长度与接点数
```

### 3 · 声纹比对（选用）

同一位主持人贯穿整份清单时，可以让来宾素材优先、主持人优先度最低。

不需要人工标记样本 —— 主持人**每一集都出现，来宾只出现在自己那一集**，
所以跨影片反复出现的声纹就是主持人：

```bash
soloclip -c configs/interviews.yaml diarize      # 先做出声纹
soloclip -c configs/interviews.yaml host         # 找出主持人
soloclip -c configs/interviews.yaml diarize --retarget   # 套用，不吃 GPU
```

只有在来宾既没特写也没乾淨音讯时，才会退回使用主持人。
不启用时（`host.min_videos` 不满足或没有 profile）就是单纯选讲最多话的人。

### 4 · 切割拼凑

1. 求「乾淨区间」＝ 主讲者语音 − 他人语音（前后各扩 `overlap_pad`）∩ 人脸通过的区间
2. 优先找**最长的单一连续段**，够长就直接取用，0 个接点
3. 不够才拼接，每段 ≥ `min_piece_seconds`，接点 ≤ `max_joins`，
   评分为「长度 − 接点罚分 − 时间间距罚分」，所以会选时间上相邻的片段
4. 所有切点对齐 ASR 词界／句界；`max_seconds` 的余裕是留给「把话讲完」的
5. 凑不到 `min_seconds` 就判失败不输出 —— 宁可少产出，不要产出不合格素材

影像硬切、音讯接点 20ms 微淡接、整支跑一次 loudnorm。

### 5 · 转档（选用）

```bash
soloclip -c configs/interviews.yaml pair-audio   # 每支成品的对应音档
```

抽的是成品本身的音轨（stream copy），不是重新选段 —— pair 的意义在于两半是同一个瞬间。

---

## 安装

```bash
conda create -n soloclip python=3.11 -y && conda activate soloclip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"                       # 之后就能直接用 soloclip 指令
conda install -c conda-forge nodejs -y        # YouTube 的 JS 签章挑战需要

# insightface 会把 CPU 版 onnxruntime 装回来、盖掉 GPU build：
pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```

### HuggingFace token（必要）

`pyannote/speaker-diarization-3.1` 与 `pyannote/segmentation-3.0` 都是 gated model，
**必须在 HF 网页上逐一按同意**，光有 token 不够（未同意会拿到 403 GatedRepoError）。
这是唯一需要使用者手动介入的前置步骤。

token 来源依序是：`HF_TOKEN` 环境变数 → `.env` → `~/.cache/huggingface/token`
（即 `huggingface-cli login` 留下的），所以已经登入过 HF CLI 的机器不必另外设定。

### 已知的版本地雷

这些组合都已经在 `pyproject.toml` 与程式里处理好了，
**改动依赖之前先看这张表** —— 其中好几条的失败是静默的。

| 症状 | 原因与对策 |
| --- | --- |
| ASD 慢十倍，stderr 有 `libcublasLt.so.13 not found` | onnxruntime-gpu ≥1.23 是对 CUDA 13 编的，torch 带的是 CUDA 12，会**静默**掉回 CPU。钉 `onnxruntime-gpu==1.22.0` |
| insightface 装完 CUDA provider 消失 | insightface 依赖 CPU 版 `onnxruntime`，同名套件互相复盖。移除 CPU 版后 `--force-reinstall --no-deps onnxruntime-gpu` |
| `hf_hub_download() got an unexpected keyword argument 'use_auth_token'` | pyannote 3.4 仍用旧参数，huggingface_hub 1.0 已移除。钉 `huggingface_hub<1.0` |
| HF 下载卡在 0 bytes（但 curl 正常） | `hf-xet` 传输后端。`config.py` 载入时设 `HF_HUB_DISABLE_XET=1` |
| `UnpicklingError: Weights only load failed` | torch 2.6 把 `torch.load` 预设改为 `weights_only=True`。`diarize._torch_load_patch()` 在载入 pipeline 期间限缩范围地还原 |
| `Requested format is not available` / 只拿得到 storyboard | **缺 JS runtime + EJS 解题脚本**，两者缺一就拿不到任何真实格式。装 node 并设 `download.remote_components: "ejs:github"`。这条坑了 41 支影片，且讯息会伪装成 SABR 或「只有 180p」——那 180 其实是缩图高度 |
| `Sign in to confirm you're not a bot` | 需要登入过的 cookie。`download.cookies_from_browser`（WSL 只有 Firefox 可用，Chrome/Edge 的 cookie 由 DPAPI 加密解不开） |
| `The page needs to be reloaded` | 换 player client：`download.youtube_player_client: "web_safari,android"` |

**cookie、player client、JS runtime 三者都必须同时套用在 probe 与 download 两个入口**
（`download.probe_video_id()` 用的是另一个乾淨的 `YoutubeDL`），只改下载端会漏掉。


---

## 用法

```bash
soloclip -c configs/talks.yaml run                 # 整份清单跑到底
soloclip -c configs/talks.yaml status              # 每支影片的阶段进度
make up   CFG=configs/talks.yaml                   # 背景执行，含崩溃自动重启
make status CFG=configs/talks.yaml
```

`-c` 是全域选项，要放在子指令**前面**。

除错时每个阶段都能单独重跑：

```bash
soloclip -c configs/talks.yaml asd    --video-id=-ABCDEFGHIJ --force
soloclip -c configs/talks.yaml select --video-id=-ABCDEFGHIJ --force
```

video id 以 `-` 开头时要用等号形式，否则 argparse 会当成旗标。

---

## 设定档

`configs/base.yaml` 放共用的门槛，各清单只写自己的差异：

```yaml
extends: base.yaml

paths:
  url_list: url_list_interviews.txt
  work_dir: work_interviews
```

| 档案 | 用途 |
| --- | --- |
| `configs/base.yaml` | 共用门槛与行为 |
| `configs/talks.yaml` | 演讲／单人简报 |
| `configs/interviews.yaml` | 访谈节目，启用主持人优先序 |
| `configs/podcasts.yaml` | 纯音讯来源，跳过人脸阶段 |

---

## 专案结构

```
src/soloclip/     download → audio → diarize → asr → asd → select → render
configs/          base.yaml + 各清单的差异
tools/            supervise/watchdog（长时间执行）、sweep（门槛扫描）、rate（吞吐量）
tests/            pytest（`pytest` 直接跑）
```

产生出来的东西不放在 repo 里。落点由 `paths.data_root` 决定，预设是 `".."`
（clone 下来那个目录的上一层），所以一份典型的工作目录长这样：

```
你的工作目录/
├── soloclip/             ← clone 下来的 repo，版控只涵盖这里
│   ├── src/ configs/ tools/ tests/
│   └── README.md LICENSE pyproject.toml Makefile
├── url_list*.txt         ← 你的清单
├── work*/                ← 中间产物（实测会长到 30GB）
├── out*/ out*_audio/     ← 成品
└── logs*/ var/
```

环境变数 `SOLOCLIP_DATA` 优先于设定档，方便同一份设定跑在不同储存配置的机器上：

```bash
export SOLOCLIP_DATA=/mnt/bigdisk/soloclip
soloclip -c configs/talks.yaml status
```

阶段纪录里的路径是相对于 data_root 存的，所以整个资料目录可以搬走；
搬完若还改了里面的目录名，跑一次 `tools/relocate.py --map 旧=新` 即可。

`work*/cache/` 是 ASD 的逐帧测量值，删掉就得重付 GPU；调门槛前别清。

各项门槛为何是现在这个数字，都写在 `configs/base.yaml` 的註解里；
长时间执行（崩溃自动重启、存活监看、多份清单依序处理）看 `tools/` 与 `make help`。

---

## 授权

MIT（见 `LICENSE`）。

程式码本身是 MIT，但它处理的**素材不是**：下载下来的影片、剪出来的片段，着作权都属于原作者。这个工具不会、也无法
赋予你散布那些内容的权利。`.gitignore` 把它们全部挡在版控之外，一部分正是
为了避免不小心连同素材一起散布。
