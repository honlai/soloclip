# soloclip

**solo** = 畫面裡只有一個人、聲音裡也只有一個人。

給一份影片清單，自動剪出**每部一支、20 秒左右、同一個人正對鏡頭說話、且沒有其他人聲**的片段。
畫面不合格但聲音乾淨時，退而輸出同樣長度的純音訊。

```
清單  →  人臉辨識  →  聲紋比對(選用)  →  切割拼湊  →  轉檔(選用)
```

實跑成績：355 支網址 → 248 支成品（影片 217、純音訊 31），
其中 127 支（51%）是完全沒有接點的單一連續鏡頭。

English: [README.md](README.md)
简体中文说明：[README.zh-cn.md](README.zh-cn.md)

---

## 五個步驟

### 1 · 給清單

一行一個網址，放進 `url_list*.txt`。設定檔指定要讀哪些：

```yaml
paths:
  url_list:
    - ../url_list.txt
    - ../url_list_2.txt      # 多份清單會依 video id 去重
```

新增清單不會重跑舊的：已經產出成品的影片會被跳過。

### 2 · 人臉辨識

對音訊乾淨的區間抽幀，逐幀判斷四件事，全部通過才算「這個人正在對鏡頭說話」：

| 條件 | 判準 | 參數 |
| --- | --- | --- |
| 臉夠大 | 臉寬佔畫面比例 | `asd.min_face_ratio` |
| 正對鏡頭 | 頭部 yaw / pitch 各自設限 | `asd.max_yaw_deg` / `max_pitch_deg` |
| 是同一個人 | 臉部 embedding 餘弦距離 | `asd.face_id_threshold` |
| 真的在說話 | 對齊後嘴部區域的幀差能量 | `asd.lip_motion_min` |

身分基準不預設「最大的臉就是主角」，而是取主講者說話期間所有臉的最大群集中心，
所以旁邊有觀眾入鏡也不會選錯人。

**逐幀原始測量值會存成 `.npz`**，調門檻不必重跑 GPU：

```bash
soloclip -c configs/talks.yaml asd --rescore     # 套用新門檻，秒級
python tools/sweep.py --yaw 30 35 40 --size 0.06 0.05      # 掃描並回報實際成品長度與接點數
```

### 3 · 聲紋比對（選用）

同一位主持人貫穿整份清單時，可以讓來賓素材優先、主持人優先度最低。

不需要人工標記樣本 —— 主持人**每一集都出現，來賓只出現在自己那一集**，
所以跨影片反覆出現的聲紋就是主持人：

```bash
soloclip -c configs/interviews.yaml diarize      # 先做出聲紋
soloclip -c configs/interviews.yaml host         # 找出主持人
soloclip -c configs/interviews.yaml diarize --retarget   # 套用，不吃 GPU
```

只有在來賓既沒特寫也沒乾淨音訊時，才會退回使用主持人。
不啟用時（`host.min_videos` 不滿足或沒有 profile）就是單純選講最多話的人。

### 4 · 切割拼湊

1. 求「乾淨區間」＝ 主講者語音 − 他人語音（前後各擴 `overlap_pad`）∩ 人臉通過的區間
2. 優先找**最長的單一連續段**，夠長就直接取用，0 個接點
3. 不夠才拼接，每段 ≥ `min_piece_seconds`，接點 ≤ `max_joins`，
   評分為「長度 − 接點罰分 − 時間間距罰分」，所以會選時間上相鄰的片段
4. 所有切點對齊 ASR 詞界／句界；`max_seconds` 的餘裕是留給「把話講完」的
5. 湊不到 `min_seconds` 就判失敗不輸出 —— 寧可少產出，不要產出不合格素材

影像硬切、音訊接點 20ms 微淡接、整支跑一次 loudnorm。

### 5 · 轉檔（選用）

```bash
soloclip -c configs/interviews.yaml pair-audio   # 每支成品的對應音檔
```

抽的是成品本身的音軌（stream copy），不是重新選段 —— pair 的意義在於兩半是同一個瞬間。

---

## 安裝

```bash
conda create -n soloclip python=3.11 -y && conda activate soloclip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"                       # 之後就能直接用 soloclip 指令
conda install -c conda-forge nodejs -y        # YouTube 的 JS 簽章挑戰需要

# insightface 會把 CPU 版 onnxruntime 裝回來、蓋掉 GPU build：
pip uninstall -y onnxruntime && pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```

### HuggingFace token（必要）

`pyannote/speaker-diarization-3.1` 與 `pyannote/segmentation-3.0` 都是 gated model，
**必須在 HF 網頁上逐一按同意**，光有 token 不夠（未同意會拿到 403 GatedRepoError）。
這是唯一需要使用者手動介入的前置步驟。

token 來源依序是：`HF_TOKEN` 環境變數 → `.env` → `~/.cache/huggingface/token`
（即 `huggingface-cli login` 留下的），所以已經登入過 HF CLI 的機器不必另外設定。

### 已知的版本地雷

這些組合都已經在 `pyproject.toml` 與程式裡處理好了，
**改動依賴之前先看這張表** —— 其中好幾條的失敗是靜默的。

| 症狀 | 原因與對策 |
| --- | --- |
| ASD 慢十倍，stderr 有 `libcublasLt.so.13 not found` | onnxruntime-gpu ≥1.23 是對 CUDA 13 編的，torch 帶的是 CUDA 12，會**靜默**掉回 CPU。釘 `onnxruntime-gpu==1.22.0` |
| insightface 裝完 CUDA provider 消失 | insightface 依賴 CPU 版 `onnxruntime`，同名套件互相覆蓋。移除 CPU 版後 `--force-reinstall --no-deps onnxruntime-gpu` |
| `hf_hub_download() got an unexpected keyword argument 'use_auth_token'` | pyannote 3.4 仍用舊參數，huggingface_hub 1.0 已移除。釘 `huggingface_hub<1.0` |
| HF 下載卡在 0 bytes（但 curl 正常） | `hf-xet` 傳輸後端。`config.py` 載入時設 `HF_HUB_DISABLE_XET=1` |
| `UnpicklingError: Weights only load failed` | torch 2.6 把 `torch.load` 預設改為 `weights_only=True`。`diarize._torch_load_patch()` 在載入 pipeline 期間限縮範圍地還原 |
| `Requested format is not available` / 只拿得到 storyboard | **缺 JS runtime + EJS 解題腳本**，兩者缺一就拿不到任何真實格式。裝 node 並設 `download.remote_components: "ejs:github"`。這條坑了 41 支影片，且訊息會偽裝成 SABR 或「只有 180p」——那 180 其實是縮圖高度 |
| `Sign in to confirm you're not a bot` | 需要登入過的 cookie。`download.cookies_from_browser`（WSL 只有 Firefox 可用，Chrome/Edge 的 cookie 由 DPAPI 加密解不開） |
| `The page needs to be reloaded` | 換 player client：`download.youtube_player_client: "web_safari,android"` |

**cookie、player client、JS runtime 三者都必須同時套用在 probe 與 download 兩個入口**
（`download.probe_video_id()` 用的是另一個乾淨的 `YoutubeDL`），只改下載端會漏掉。


---

## 用法

```bash
soloclip -c configs/talks.yaml run                 # 整份清單跑到底
soloclip -c configs/talks.yaml status              # 每支影片的階段進度
make up   CFG=configs/talks.yaml                   # 背景執行，含崩潰自動重啟
make status CFG=configs/talks.yaml
```

`-c` 是全域選項，要放在子指令**前面**。

除錯時每個階段都能單獨重跑：

```bash
soloclip -c configs/talks.yaml asd    --video-id=-ABCDEFGHIJ --force
soloclip -c configs/talks.yaml select --video-id=-ABCDEFGHIJ --force
```

video id 以 `-` 開頭時要用等號形式，否則 argparse 會當成旗標。

---

## 設定檔

`configs/base.yaml` 放共用的門檻，各清單只寫自己的差異：

```yaml
extends: base.yaml

paths:
  url_list: url_list_interviews.txt
  work_dir: work_interviews
```

| 檔案 | 用途 |
| --- | --- |
| `configs/base.yaml` | 共用門檻與行為 |
| `configs/talks.yaml` | 演講／單人簡報 |
| `configs/interviews.yaml` | 訪談節目，啟用主持人優先序 |
| `configs/podcasts.yaml` | 純音訊來源，跳過人臉階段 |

---

## 專案結構

```
src/soloclip/     download → audio → diarize → asr → asd → select → render
configs/          base.yaml + 各清單的差異
tools/            supervise/watchdog（長時間執行）、sweep（門檻掃描）、rate（吞吐量）
tests/            pytest（`pytest` 直接跑）
```

產生出來的東西不放在 repo 裡。落點由 `paths.data_root` 決定，預設是 `".."`
（clone 下來那個目錄的上一層），所以一份典型的工作目錄長這樣：

```
你的工作目錄/
├── soloclip/             ← clone 下來的 repo，版控只涵蓋這裡
│   ├── src/ configs/ tools/ tests/
│   └── README.md LICENSE pyproject.toml Makefile
├── url_list*.txt         ← 你的清單
├── work*/                ← 中間產物（實測會長到 30GB）
├── out*/ out*_audio/     ← 成品
└── logs*/ var/
```

環境變數 `SOLOCLIP_DATA` 優先於設定檔，方便同一份設定跑在不同儲存配置的機器上：

```bash
export SOLOCLIP_DATA=/mnt/bigdisk/soloclip
soloclip -c configs/talks.yaml status
```

階段紀錄裡的路徑是相對於 data_root 存的，所以整個資料目錄可以搬走；
搬完若還改了裡面的目錄名，跑一次 `tools/relocate.py --map 舊=新` 即可。

`work*/cache/` 是 ASD 的逐幀測量值，刪掉就得重付 GPU；調門檻前別清。

各項門檻為何是現在這個數字，都寫在 `configs/base.yaml` 的註解裡；
長時間執行（崩潰自動重啟、存活監看、多份清單依序處理）看 `tools/` 與 `make help`。

---

## 授權

MIT（見 `LICENSE`）。

程式碼本身是 MIT，但它處理的**素材不是**：下載下來的影片、剪出來的片段，著作權都屬於原作者。這個工具不會、也無法
賦予你散布那些內容的權利。`.gitignore` 把它們全部擋在版控之外，一部分正是
為了避免不小心連同素材一起散布。
