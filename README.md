# Moshi LoRA Fine-tuning

`kyutai-labs/moshi-finetune` を使い、`llm-jp/llm-jp-moshi-v1` を LoRA で追加学習するための作業リポジトリです。Python 実行はすべて `uv` 経由に統一しています。

参考:

- `moshi-finetune`: https://github.com/kyutai-labs/moshi-finetune
- `llm-jp/llm-jp-moshi-v1`: https://huggingface.co/llm-jp/llm-jp-moshi-v1

## 目次

- [まずこれ](#まずこれ)
- [全体像](#全体像)
- [1. 環境構築](#1-環境構築)
- [2. モデル取得](#2-モデル取得)
- [3. .env 設定](#3-env-設定)
- [4. 音声素材の用意](#4-音声素材の用意)
- [5. 前処理ツール](#5-前処理ツール)
- [6. raw から stereo を作る](#6-raw-から-stereo-を作る)
- [7. JSONL と学習用 transcript](#7-jsonl-と学習用-transcript)
- [8. 学習](#8-学習)
- [9. 推論確認](#9-推論確認)
- [10. 困った時](#10-困った時)

---

## まずこれ

最短で流れを確認する場合は、以下の順に進めます。

```bash
# 1. 環境構築
bash scripts/bootstrapMoshiEnv.sh

# 2. モデル取得
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune python scripts/downloadModel.py

# 3. .env 作成
cp .env.example .env

# 4. raw 音声を確認
uv run --project moshi-finetune python scripts/processRawToStereo.py --dry-run

# 5. 少量だけ試作
uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --limit 1 \
  --max-segments 10

# 6. 問題なければ本番変換
uv run --project moshi-finetune python scripts/processRawToStereo.py

# 7. JSONL と学習用文字起こし
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py
uv run --project moshi-finetune python scripts/annotateDataset.py datasets/train.jsonl --lang ja

# 8. 学習
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node 1 \
  moshi-finetune/train.py \
  config/llmJpMoshiLora.yaml
```

---

## 全体像

```text
raw音声
  ↓
必要なら UVR5 / Demucs / 無音削除
  ↓
一人配信なら Whisper + OpenAI互換API + TTS で右ch補完
  ↓
datasets/stereo/*.wav
  ↓
datasets/train.jsonl
  ↓
annotateDataset.py で学習用 transcript json 生成
  ↓
LoRA 学習
  ↓
loras/llmJpMoshiV1/latest/
```

主な保存先:

| パス | 用途 |
|---|---|
| `models/` | Hugging Face モデル、Whisper モデル、キャッシュ |
| `datasets/raw/` | 元音声、UVR5 などで切り抜いた声素材 |
| `datasets/cache/` | Whisper / LLM / TTS の途中結果 |
| `datasets/tts/` | 手動TTSや確認用TTS |
| `datasets/stereo/` | 学習に使う2ch音声 |
| `datasets/train.jsonl` | 学習対象wav一覧 |
| `loras/` | 学習結果とLoRAアダプター |

補助ツールは `rich` で、処理中ファイル、完了数、経過時間、残り時間の目安を表示します。長い処理では、必ず `--dry-run` や `--limit` で小さく試してから本番処理してください。

---

## 1. 環境構築

確認:

```bash
nvidia-smi
uv --version
```

`uv` がない場合:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

依存関係を同期:

```bash
bash scripts/bootstrapMoshiEnv.sh
```

補足:

- `openai-whisper` のビルドで `pkg_resources` が見つからない問題を避けるため、`moshi-finetune/pyproject.toml` でビルド時の `setuptools` を `69.5.1` に制約しています
- 以降の Python 実行は `uv run --project moshi-finetune ...` を使います

---

## 2. モデル取得

モデル本体と Hugging Face キャッシュは `models/` に置きます。

```bash
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune python scripts/downloadModel.py
```

学習設定では `config/llmJpMoshiLora.yaml` の `hf_repo_id` に `llm-jp/llm-jp-moshi-v1` を指定しています。

---

## 3. .env 設定

一人配信から会話相手の応答を作る場合、OpenAI API 互換エンドポイントを使います。

```bash
cp .env.example .env
```

`.env`:

```dotenv
OPENAI_BASE_URL=https://litellm.kuwa.dev/v1
OPENAI_API_KEY=ここにAPIキー
OPENAI_MODEL=anthropic/claude-sonnet-4.6
KUWA_TTS_URL=https://api.kuwa.app/v1/capcut/synthesize
KUWA_TTS_TYPE=10
```

疎通確認だけしたい場合は、OpenAI Python ライブラリ経由で簡単に確認できます。

```bash
uv run --project moshi-finetune python - <<'PY'
import os
from pathlib import Path
from openai import OpenAI

for line in Path(".env").read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://litellm.kuwa.dev/v1"),
)
res = client.chat.completions.create(
    model=os.environ.get("OPENAI_MODEL", "anthropic/claude-sonnet-4.6"),
    messages=[{"role": "user", "content": "日本語でOKとだけ返してください。"}],
    max_tokens=16,
)
print(res.choices[0].message.content)
PY
```

---

## 4. 音声素材の用意

### 基本方針

音声の切り抜きやボーカル抽出は、PC 側で UVR5 などの専用ツールを使う運用がおすすめです。声だけになったファイルを `datasets/raw/` に入れてください。耳で確認しながら不要区間を削れるため、学習データの品質を上げやすいです。

おすすめの素材:

| 観点 | 内容 |
|---|---|
| 話し方 | 目的の話者が自然に話している |
| 音質 | マイク音質、音量、部屋鳴りがある程度そろっている |
| ノイズ | BGM、効果音、ゲーム音、通知音が小さい |
| 会話性 | 相槌、笑い、言い淀み、聞き返しが自然に含まれる |
| 内容 | 学習後に使いたい話題や口調に近い |

避けたい素材:

- BGM や環境音が声と同じくらい大きいもの
- 複数人が同時に長く話しているもの
- 音割れ、強いリバーブ、ノイズ除去のかけすぎ
- 読み上げ、台本、歌、叫び声が多く普段の会話と違いすぎるもの
- 権利や同意が確認できない音声

データ量の目安:

| 目的 | 目安 |
|---|---:|
| 動作確認 | 3〜10分 |
| 声や口調の軽い確認 | 30分〜2時間 |
| ある程度安定した LoRA | 5〜20時間 |
| 口調や反応まで寄せる | 20時間以上 |

量より品質が大事です。小さいけれどきれいなデータセットは、大きいけれど荒れたデータセットより安定しやすいです。

### チャンネル設計

`moshi-finetune` はステレオ wav を前提にしています。

| チャンネル | 内容 |
|---|---|
| 左ch | Moshi に学習させたい側の音声 |
| 右ch | ユーザー、聞き手、会話相手側の音声 |

一人配信では、左に元配信者音声、右に LLM + TTS で作った相槌や質問を入れます。コラボや通話音声では、話者分離して学習したい声を左、それ以外を右に寄せます。左右は途中で入れ替えないでください。

### クレンジング

やること:

- 音割れしている区間を削除する
- 長い無音、待機画面、離席、読み込み待ちを削除する
- BGM や効果音が大きい区間を削除する
- 聞き取れない同時発話を削除する
- 配信冒頭や終了画面など、学習したい会話と違う定型部分を削除する
- 個人情報や学習させたくない話題を削除する
- 音量を大きすぎず小さすぎない範囲にそろえる

やりすぎないこと:

- 強いノイズ除去で声を金属っぽくしない
- 無音を完全に詰めすぎない
- 笑い、息継ぎ、短い間を全部消さない
- すべての発話を同じ音量に潰しすぎない

---

## 5. 前処理ツール

### 無音区間を詰める

長い無音だけを削り、短い自然な間は残します。

```bash
uv run --project moshi-finetune python scripts/trimSilence.py \
  --input datasets/raw/source001.wav \
  --output datasets/raw/trimmed/source001.wav
```

フォルダ一括:

```bash
uv run --project moshi-finetune python scripts/trimSilence.py \
  --input datasets/raw/demucsVocals \
  --output datasets/raw/trimmed \
  --recursive
```

| オプション | 目安 | 意味 |
|---|---:|---|
| `--threshold-db` | `-45` | これより小さい音を無音扱い |
| `--min-silence-sec` | `0.8` | この秒数以上の無音だけ削る |
| `--keep-silence-sec` | `0.25` | 削った前後に残す余白 |
| `--min-voice-sec` | `0.15` | 短すぎるノイズを発話扱いしない |

### Demucs で声を抽出する

基本は UVR5 推奨ですが、サーバ上で簡易的にボーカル抽出したい場合は Demucs も使えます。追加依存が重いので、必要な時だけ `--with demucs` で実行します。

```bash
uv run --project moshi-finetune --with demucs python scripts/extractVocalsDemucs.py \
  --input datasets/raw/source001.wav \
  --output-dir datasets/raw/demucsVocals
```

フォルダ一括:

```bash
uv run --project moshi-finetune --with demucs python scripts/extractVocalsDemucs.py \
  --input datasets/raw/originals \
  --output-dir datasets/raw/demucsVocals
```

GPU メモリ不足時:

```bash
uv run --project moshi-finetune --with demucs python scripts/extractVocalsDemucs.py \
  --input datasets/raw/source001.wav \
  --output-dir datasets/raw/demucsVocals \
  --segment 7.8
```

出力例:

```text
datasets/raw/demucsVocals/source001_vocals.wav
```

Demucs は `--two-stems vocals` でボーカルと伴奏を分けますが、話者分離ツールではありません。複数人の声を人物ごとに分けたい場合は、手作業確認や話者分離ツールを併用してください。

---

## 6. raw から stereo を作る

### 一括処理

`datasets/raw/` に置いた一人配信音声を、ファイル名順に `datasets/stereo/` へ変換します。

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py
```

まず対象だけ確認:

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py --dry-run
```

少量だけ試作:

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --limit 1 \
  --max-segments 10 \
  --max-response-chars 40 \
  --response-delay-sec 0.5 \
  --tts-speed 1.2
```

### 途中保存と再開

途中結果は `datasets/cache/<音声ファイル名>/` に保存されます。

```text
datasets/cache/solo001/
  transcript.json        # Whisper の文字起こし結果
  responses.json         # LLM 応答と配置タイミング
  tts/
    raw/
      response0000.wav   # TTS APIの元音声
    response0000.wav     # 速度調整後の音声
```

処理が途中で落ちても、次回実行時は保存済みの `transcript.json`、`responses.json`、TTS音声を再利用します。Whisper は特に時間がかかるため、通常は `--refresh-transcript` を付けずに再実行してください。

`--max-segments` は処理対象を一時的に絞るだけで、`transcript.json` には全体の Whisper 結果を保存します。短い試作のあと、本番実行で Whisper をやり直す必要はありません。

### 応答生成の考え方

Whisper の発話間隔を見て挙動を変えます。

| 状況 | 挙動 |
|---|---|
| 発話間隔が `--merge-gap-sec` 以下 | 複数発話をまとめて1つの返答を生成 |
| 次の発話までが短い | 短い相槌寄り |
| 次の発話まで余裕がある | しっかり返答 |
| TTS音声 | デフォルトでピッチを変えずに `1.2` 倍速 |

主なオプション:

| オプション | 既定値 | 用途 |
|---|---:|---|
| `--input-dir` | `datasets/raw` | 入力音声フォルダ |
| `--output-dir` | `datasets/stereo` | ステレオ wav 出力先 |
| `--cache-dir` | `datasets/cache` | Whisper/LLM/TTS の途中結果 |
| `--dry-run` | なし | 対象ファイル一覧だけ表示 |
| `--limit` | なし | 先頭 N ファイルだけ処理 |
| `--max-segments` | なし | 各音声の先頭 N 発話だけ処理 |
| `--merge-gap-sec` | `0.8` | 近い発話をまとめる秒数 |
| `--short-gap-sec` | `2.0` | 次発話までが短いと判断する秒数 |
| `--long-gap-sec` | `5.0` | しっかり返答できると判断する秒数 |
| `--min-response-chars` | `12` | 短い相槌の最小文字数 |
| `--max-response-chars` | `48` | 返答の最大文字数 |
| `--response-delay-sec` | `0.35` | 発話終了から応答開始までの待ち時間 |
| `--response-margin-sec` | `0.25` | 次発話にかぶらないための余白 |
| `--tts-chars-per-sec` | `8.0` | 返答文字数を決めるための読み上げ速度目安 |
| `--tts-speed` | `1.2` | ピッチ維持の速度変更。`1.0` で無効 |
| `--refresh-transcript` | なし | Whisper 結果を作り直す |
| `--refresh-responses` | なし | LLM/TTS 結果を作り直す |
| `--continue-on-error` | なし | 1ファイル失敗しても次へ進む |

### 単発で stereo を作る

左右の音声が既に分かれている場合:

```bash
uv run --project moshi-finetune python scripts/makeStereoPair.py \
  --left datasets/raw/moshiSide.wav \
  --right datasets/raw/userSide.wav \
  --output datasets/stereo/sample001.wav
```

TTS だけ試す場合:

```bash
uv run --project moshi-finetune python scripts/synthesizeTts.py \
  "こんにちは" \
  --type 10 \
  --output datasets/tts/response001.wav
```

---

## 7. JSONL と学習用 transcript

`datasets/stereo/*.wav` から `datasets/train.jsonl` を作ります。

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl
```

形式:

```json
{"path": "datasets/stereo/sample001.wav", "duration": 12.34}
```

学習用 transcript を生成します。Whisper はローカル実行で、基本は `large-v3` を使います。

```bash
uv run --project moshi-finetune python scripts/annotateDataset.py \
  datasets/train.jsonl \
  --lang ja \
  --whisper-model large-v3
```

wav と json が揃っているか確認:

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl \
  --require-transcript
```

---

## 8. 学習

設定ファイル:

```text
config/llmJpMoshiLora.yaml
```

主な設定:

| 項目 | 値 |
|---|---|
| `moshi_paths.hf_repo_id` | `llm-jp/llm-jp-moshi-v1` |
| `run_dir` | `loras/llmJpMoshiV1` |
| `lora.rank` | `128` |
| `duration_sec` | `60` |
| `batch_size` | `1` |
| `param_dtype` | `float16` |

1 GPU:

```bash
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node 1 \
  moshi-finetune/train.py \
  config/llmJpMoshiLora.yaml
```

8 GPU:

```bash
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node 8 \
  --master_port 29501 \
  moshi-finetune/train.py \
  config/llmJpMoshiLora.yaml
```

LoRA は以下に保存されます。

```text
loras/llmJpMoshiV1/checkpoints/checkpoint_XXXXXX/consolidated/
```

最新 LoRA を固定パスにコピー:

```bash
uv run --project moshi-finetune python scripts/exportLatestLora.py
```

出力:

```text
loras/llmJpMoshiV1/latest/lora.safetensors
loras/llmJpMoshiV1/latest/config.json
```

---

## 9. 推論確認

```bash
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune python -m moshi.server \
  --hf-repo llm-jp/llm-jp-moshi-v1 \
  --lora-weight loras/llmJpMoshiV1/latest/lora.safetensors \
  --config-path loras/llmJpMoshiV1/latest/config.json
```

起動後、`http://localhost:8998` にアクセスします。エコーを避けるため、確認時はイヤホンかヘッドホンを使ってください。

---

## 10. 困った時

### 学習で OOM する

- `batch_size` を `1` にする
- `duration_sec` を `30` に下げる
- `gradient_checkpointing: true` を維持する

### 声質が乗りにくい

- データ量を増やす
- `max_steps` を `2000` 以上にする
- 早い段階と遅い段階のチェックポイントを聞き比べる

### 日本語応答が崩れる

- `optim.lr` を `1e-6` に下げる
- `max_steps` を増やしすぎない
- 元データの transcript JSON を確認する

### TTS API が 403 を返す

- `scripts/synthesizeTts.py` と `scripts/generateSoloConversationDataset.py` は `User-Agent` と `Accept` ヘッダを付けるように対応済みです
- 古い状態で失敗した場合は最新のスクリプトで再実行してください
- API の返却音声は拡張子が `.wav` でも内部形式が圧縮音声として判定されることがありますが、`torchaudio` で読める場合は後続処理で使えます

### LLM API が 403 を返す

- `.env` の `OPENAI_API_KEY` が正しいか確認してください
- `OPENAI_MODEL=anthropic/claude-sonnet-4.6` が利用可能か確認してください
- 途中まで成功していれば `datasets/cache/<音声名>/responses.json` に保存されているため、設定修正後に同じコマンドを再実行できます

---

## 注意点

`llm-jp/llm-jp-moshi-v1` は Apache-2.0 で公開されていますが、モデルカードでは試作段階であること、出力に不自然さや不適切な内容が含まれる可能性、悪意ある利用を意図しないことが明記されています。公開や配布を行う場合は、元モデルと追加学習データのライセンス、話者の同意、音声の利用範囲を必ず確認してください。
