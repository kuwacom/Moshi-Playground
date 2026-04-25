## LoRA カスタム化 完全手順

---

## Step 0：環境確認

```bash
# CUDA確認
nvidia-smi
# → P40 × 8 が見えていればOK

# Python確認（3.10以上必要）
python3 --version

# 推奨：uv を使う（pip より10倍速い）
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 1：リポジトリのセットアップ

```bash
git clone git@github.com:kyutai-labs/moshi-finetune.git
cd moshi-finetune

# uv を使う場合（推奨）
# 以降のコマンドは uv run を prefix するだけでOK
# 明示的にインストールしたい場合は：
uv sync

# pip を使う場合
pip install -e .
```

---

## Step 2：データ収集・前処理パイプライン

### 2-1：音声ファイルの準備

```
data/
├── raw/                    ← 元の音声（コラボ動画から抽出など）
│   ├── collab_01.wav
│   └── collab_02.wav
├── stereo/                 ← 2ch化した音声（学習に使う）
│   ├── collab_01.wav
│   ├── collab_01.json      ← トランスクリプト（自動生成）
│   └── ...
└── dataset.jsonl           ← ファイルリスト
```

### 2-2：モノラル音声を2ch化（コラボ回の場合）

```python
# separate_speakers.py
# pyannote で話者分離 → ステレオ2ch化

from pyannote.audio import Pipeline
import soundfile as sf
import numpy as np

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

def mono_to_stereo(input_path, output_path):
    # 話者分離
    diarization = pipeline(input_path)
    
    audio, sr = sf.read(input_path)
    ch0 = np.zeros_like(audio)  # Moshi側（学習させたい声）
    ch1 = np.zeros_like(audio)  # ユーザー側
    
    speakers = list(set([s for _, _, s in diarization.itertracks(yield_label=True)]))
    target_speaker = speakers[0]  # 配信者の声を0chに
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = int(turn.start * sr)
        end = int(turn.end * sr)
        if speaker == target_speaker:
            ch0[start:end] = audio[start:end]
        else:
            ch1[start:end] = audio[start:end]
    
    stereo = np.stack([ch0, ch1], axis=1)
    sf.write(output_path, stereo, sr)

mono_to_stereo("raw/collab_01.wav", "stereo/collab_01.wav")
```

### 2-3：ソロ配信の場合（LLM + TTS で右ch生成）

```python
# generate_right_ch.py
# Whisper でテキスト化 → LLM で応答生成 → TTS で音声化 → 右chに配置

import whisper
from openai import OpenAI
import soundfile as sf
import numpy as np

# Step A: 配信音声をテキスト化
model = whisper.load_model("large")
result = model.transcribe("raw/solo_01.wav", language="ja")
segments = result["segments"]

# Step B: 各発話に対してLLMで応答生成
client = OpenAI()

def generate_response(text):
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "あなたは自然な会話をする人物です。相手の発言に対して短く自然に返答してください。"},
            {"role": "user", "content": text}
        ]
    )
    return res.choices[0].message.content

# Step C: TTS で音声化（Style-BERT-VITS2 推奨）
# → 生成した応答テキストを音声化して右chに配置
```

### 2-4：トランスクリプト生成（WhisperX）

```bash
# annotate.py が自動でやってくれる
pip install whisperx

# dataset.jsonl を先に作る
python3 - << 'EOF'
import sphn, json
from pathlib import Path

paths = [str(f) for f in Path("stereo").glob("*.wav")]
durations = sphn.durations(paths)

with open("dataset.jsonl", "w") as f:
    for p, d in zip(paths, durations):
        if d is None:
            continue
        json.dump({"path": p, "duration": d}, f)
        f.write("\n")
EOF

# トランスクリプト自動生成（.json ファイルが各wavと同じ場所に生成される）
python annotate.py dataset.jsonl
```

生成される `.json` の中身はこんな感じ：

```json
{
  "words": [
    {"word": "今日", "start": 0.24, "end": 0.48, "channel": 0},
    {"word": "は", "start": 0.48, "end": 0.56, "channel": 0},
    {"word": "うん", "start": 0.60, "end": 0.80, "channel": 1}
  ]
}
```

---

## Step 3：学習設定ファイルの作成

```yaml
# config/kuwa_lora.yaml

moshi_paths:
  hf_repo_id: "llm-jp/llm-jp-moshi-v1"  # LLM-jp-Moshi-v1 をベースに

run_dir: "./runs/kuwa_voice_v1"

# データ
data:
  train_data: "./dataset.jsonl"
  eval_data: null   # 評価データがあれば指定
  shuffle: true

# シーケンス長（秒）
# P40 24GB × 1枚なら 30〜50 が安全圏
# 8枚並列なら 100 まで上げられる
duration_sec: 60

# バッチサイズ
# 1枚あたり。OOMが出たら下げる
batch_size: 8

# 学習ステップ数
# 10〜30時間データなら 1000〜2000 が目安
max_steps: 1500

# LoRA 設定
lora:
  enable: true
  rank: 128          # 声質適応なら 64〜128 で十分
  scaling: 2.0
  ft_embed: false    # 埋め込み層はフルFTしない

# 最適化
optim:
  lr: 2.0e-6         # 推奨値
  weight_decay: 0.1
  pct_start: 0.05    # 5% をウォームアップに使う

# 損失の重み付け
first_codebook_weight_multiplier: 100  # セマンティックトークンを重視
text_padding_weight: 0.1               # PADトークンの損失を下げる

# メモリ節約
gradient_checkpointing: true

# チェックポイント
ckpt_freq: 200
log_freq: 10

# LoRAアダプターだけ保存（推奨）
save_adapters: true
full_finetuning: false

# W&B（任意）
# wandb:
#   key: "YOUR_KEY"
#   project: "moshi-kuwa-voice"
```

---

## Step 4：学習実行

### P40 × 1枚で試す場合（動作確認）

```bash
torchrun --nproc-per-node 1 -m train config/kuwa_lora.yaml
```

### P40 × 8枚で本番学習

```bash
torchrun \
  --nproc-per-node 8 \
  --master_port $RANDOM \
  -m train config/kuwa_lora.yaml
```

### Docker コンテナで回す場合（くわさんの環境に合わせて）

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git
RUN pip install uv

WORKDIR /workspace
COPY . .
RUN uv sync
```

```bash
docker run --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/runs:/workspace/runs \
  moshi-finetune \
  torchrun --nproc-per-node 8 -m train config/kuwa_lora.yaml
```

### 学習中のログ確認

```bash
# ステップごとの loss が出る
# Step 200: train_loss=2.34, lr=1.8e-6
# Step 400: train_loss=1.98, lr=2.0e-6
# ...

# loss が 2.0 を下回ってきたら声質が乗り始めているサイン
# 1.5 以下になると話し方まで変化してくる（データ量次第）
```

---

## Step 5：推論・動作確認

```bash
# moshi 本体のインストール
pip install moshi<=0.2.2 sphn==0.1.12

# LoRA アダプターを指定して起動
python -m moshi.server \
  --hf-repo llm-jp/llm-jp-moshi-v1 \
  --lora-weight ./runs/kuwa_voice_v1/checkpoints/checkpoint_001500/consolidated/lora.safetensors \
  --config-path ./runs/kuwa_voice_v1/checkpoints/checkpoint_001500/consolidated/config.json

# ブラウザで http://localhost:8998 にアクセス
# → Web UI でマイクから話しかけて動作確認
```

---

## Step 6：複数の「声」を管理する場合

LoRA の最大のメリットがここで活きます：

```bash
runs/
├── voice_A/lora.safetensors   ← Aさんの声
├── voice_B/lora.safetensors   ← Bさんの声
└── voice_C/lora.safetensors   ← Cさんの声

# 切り替えは --lora-weight を差し替えるだけ
python -m moshi.server \
  --hf-repo llm-jp/llm-jp-moshi-v1 \
  --lora-weight ./runs/voice_B/lora.safetensors \
  ...
```

ベースモデル（数十GB）は共有したまま、アダプター（数百MB）だけ差し替えで切り替えられます。

---

## トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| OOM（メモリ不足） | batch_size が大きい | batch_size を半分に下げる |
| OOM が続く | duration_sec が長い | duration_sec を 30 に下げる |
| loss が下がらない | lr が高すぎ | lr を 1e-6 に下げる |
| 声が変わらない | max_steps が少ない | 2000〜3000 に増やす |
| 日本語が崩れる | lr が高すぎ or steps が多すぎ | lr を下げ、早めのチェックポイントを使う |

---

## くわさんの環境での推定学習時間

```
P40 × 8枚、データ 30時間、max_steps=1500 の場合

公式実績：H100 × 8枚 → 10.7k tokens/sec
P40 は H100 の約 1/4 の性能
→ P40 × 8枚 ≈ 2.5k tokens/sec

総トークン数 = 1500 × 8 × 8 × 60 × 9 × 12.5 ≈ 6.5億トークン
推定時間 ≈ 6.5億 / 2500 ≈ 260,000秒 ≈ 72時間

# gradient_checkpointing: true にすると遅くなるが
# batch_size を上げられるのでトータルは変わらないことが多い
```

P40 は FP16 が弱いので、**BF16 ではなく FP16 で動かす設定が必要な場合がある**点だけ注意です。

---

## 全体の流れ（再掲）

```
Step 0: 環境構築（uv / pip）
Step 1: moshi-finetune clone
Step 2: データ前処理
  ├── コラボ回 → pyannote で話者分離 → ステレオ2ch化
  └── ソロ回 → Whisper → LLM応答生成 → TTS → 右ch合成
Step 3: annotate.py でトランスクリプト生成
Step 4: YAML 設定ファイル作成
Step 5: torchrun で学習
Step 6: moshi.server で推論確認
```