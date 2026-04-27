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
- [5. 完全自動でデータセットを作る](#5-完全自動でデータセットを作る)
- [6. 手動処理と各ツール](#6-手動処理と各ツール)
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

# 6. 問題なければデータセット準備を一括実行
bash prepare-dataset.sh

# 7. 学習
bash train-run.sh
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

モデル本体は Hugging Face Hub キャッシュとして `models/huggingface/hub/` に置きます。`scripts/downloadModel.py` は通常、別のフルコピーを作らず、このキャッシュだけを使います。

```bash
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune python scripts/downloadModel.py
```

学習設定では `config/llmJpMoshiLora.yaml` の `hf_repo_id` に `llm-jp/llm-jp-moshi-v1` を指定しています。

`models/llm-jp-moshi-v1/` のような直接コピーは学習には不要です。`--local-dir` を明示した場合だけ作成してください。

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

## 5. 完全自動でデータセットを作る

ここでは、`datasets/raw/` に置いた一人配信音声から、学習に使う `datasets/train.jsonl` と transcript まで一気に作る流れを説明します。まずはこの方法で全体を通し、音質や応答が気になる場合だけ次の「手動処理と各ツール」で細かく調整してください。

### 1. raw 音声を置く

PC 側で UVR5 などを使って声だけに近い状態へ切り抜いた wav を、`datasets/raw/` に入れます。

```text
datasets/raw/source001.wav
datasets/raw/source002.wav
datasets/raw/source003.wav
```

### 2. 対象ファイルを確認する

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py --dry-run
```

`--dry-run` は変換せず、処理対象になるファイルだけを表示します。意図しないファイルが混ざっていないか、ここで確認してください。

### 3. 少量だけ試作する

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --limit 1 \
  --max-segments 10 \
  --max-response-chars 40 \
  --response-delay-sec 0.5 \
  --tts-speed 1.2
```

この試作では、先頭 1 ファイルの先頭 10 発話だけを処理します。Whisper、LLM 応答、TTS、stereo 合成までまとめて確認できます。

確認する場所:

| パス | 見ること |
|---|---|
| `datasets/stereo/` | 生成された2ch wav |
| `datasets/cache/<音声ファイル名>/transcript.json` | Whisper の文字起こし |
| `datasets/cache/<音声ファイル名>/responses.json` | LLM 応答と配置タイミング |
| `datasets/cache/<音声ファイル名>/tts/` | 速度調整後の TTS 音声 |

### 4. 問題なければ本番変換する

データセット準備をまとめて実行する場合:

```bash
bash prepare-dataset.sh
```

`prepare-dataset.sh` は以下を順番に実行します。

| 処理 | 内容 |
|---|---|
| `processRawToStereo.py` | raw音声をstereo学習音声へ変換 |
| `prepareDatasetJsonl.py` | `datasets/stereo/` から `datasets/train.jsonl` を生成 |
| `annotateDataset.py` | Whisper large-v3 で学習用 transcript json を生成 |

各ステップを個別に調整したい場合は、以下のように手動で実行します。

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py
```

`datasets/raw/` の wav をファイル名順に処理し、`datasets/stereo/` へ stereo wav を出力します。途中で失敗しても `datasets/cache/` に途中結果が残るため、再実行時は保存済みの Whisper / LLM / TTS 結果を再利用します。

失敗したファイルを飛ばして続けたい場合:

```bash
uv run --project moshi-finetune python scripts/processRawToStereo.py \
  --continue-on-error
```

### 5. JSONL と transcript を作る

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl

uv run --project moshi-finetune python scripts/annotateDataset.py \
  datasets/train.jsonl \
  --lang ja \
  --whisper-model large-v3 \
  --whisper-cache-dir models/whisper
```

最後に、wav と transcript json が揃っているか確認します。

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl \
  --require-transcript
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

Whisper は特に時間がかかるため、通常は `--refresh-transcript` を付けずに再実行してください。LLM 応答や TTS だけ作り直したい場合は `--refresh-responses` を使います。

`--max-segments` は処理対象を一時的に絞るだけで、`transcript.json` には全体の Whisper 結果を保存します。短い試作のあと、本番実行で Whisper をやり直す必要はありません。

### 応答生成の考え方

Whisper の発話間隔と発話内容を見て、右チャンネルへ入れる音声を自動で切り替えます。元音声のフィラー、息継ぎ、笑い、短い間は削りません。左チャンネルの時間軸も動かさず、右チャンネルの挿入位置だけを調整します。

| 状況 | 挙動 |
|---|---|
| 発話間隔が `--merge-gap-sec` 以下 | 複数発話をまとめて1つの返答を生成 |
| コメントに答えているような発話で、直前に十分な間がある | 発話の手前に「質問コメント」を生成 |
| 独立した説明や雑談に見える発話 | 発話の後ろに「返事」を生成 |
| 次の発話までが短い | 短い相槌寄りに文字数を制限 |
| 次の発話まで余裕がある | しっかり返答 |
| 左chの長い発話で右ch比率が足りない | 発話の途中に短い相槌を追加 |
| TTS音声 | デフォルトでピッチを変えずに `1.2` 倍速 |

`--interaction-mode auto` が既定です。配信者が明らかにコメントへ返しているような場面だけ `pre_question` を選び、それ以外は `reply` として扱います。

モードを固定したい場合:

| モード | 挙動 |
|---|---|
| `--interaction-mode auto` | 手前質問コメントと返事を自動選択。手前に自然に置けない場合は返事へフォールバック |
| `--interaction-mode pre-question` | 手前質問コメントだけ生成。自然に置けない発話はスキップ |
| `--interaction-mode reply` | 発話後の返事だけ生成 |

既に `datasets/cache/<音声名>/responses.json` がある場合はキャッシュが優先されます。ただし、現在のモードと違う種類のキャッシュは再利用しません。モードを変えて全体を作り直したい時は `--refresh-responses` を付けてください。

### 左右の発話量バランス

Moshi の学習データでは、左chだけが長く続くより、右chもある程度しゃべっている方が安定しやすいです。このため既定では、`--interaction-mode auto` の時に、右chの発話量が足りなければ長い左ch発話の途中へ短い相槌を追加します。

発話量は Whisper の発話区間と生成TTSの長さから概算します。

```text
rightRatio = 右ch発話秒数 / (左ch発話秒数 + 右ch発話秒数)
```

既定では `--target-right-ratio 0.4` を目標にし、`--max-right-ratio 0.5` を超えない範囲で補完します。つまり、左ch : 右ch がだいたい `60:40` 付近になるように、長い説明の途中へ文脈に合う短い相槌や促しをLLMで生成して足します。元音声は削らず、左chの時間軸も動かしません。

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
| `--interaction-mode` | `auto` | `auto` / `reply` / `pre-question` を選択 |
| `--min-insert-gap-sec` | `0.8` | 返事を自然に置ける最小の発話間隔 |
| `--pre-question-gap-sec` | `1.6` | 手前質問コメントを検討する最小の直前間隔 |
| `--short-gap-sec` | `2.0` | 次発話までが短いと判断する秒数 |
| `--long-gap-sec` | `5.0` | しっかり返答できると判断する秒数 |
| `--min-response-chars` | `12` | 短い相槌の最小文字数 |
| `--max-response-chars` | `48` | 返答の最大文字数 |
| `--response-delay-sec` | `0.35` | 発話終了から応答開始までの待ち時間 |
| `--response-margin-sec` | `0.25` | 次発話にかぶらないための余白 |
| `--tts-chars-per-sec` | `8.0` | 返答文字数を決めるための読み上げ速度目安 |
| `--tts-speed` | `1.2` | ピッチ維持の速度変更。`1.0` で無効 |
| `--balance-fill-mode` | `auto` | `auto` は `interaction-mode auto` の時だけ比率補完。`always` で常時、`off` で無効 |
| `--target-right-ratio` | `0.4` | 目標の右ch発話比率 |
| `--max-right-ratio` | `0.5` | 右chを増やしすぎない上限 |
| `--long-turn-fill-sec` | `12.0` | この秒数以上の左ch発話を途中補完の対象にする |
| `--fill-interval-sec` | `6.0` | 長い左ch発話の中で相槌候補を置く間隔 |
| `--fill-max-chars` | `14` | 比率補完で挿入する短い発話の最大文字数 |
| `--refresh-transcript` | なし | Whisper 結果を作り直す |
| `--refresh-responses` | なし | LLM/TTS 結果を作り直す |
| `--continue-on-error` | なし | 1ファイル失敗しても次へ進む |

---

## 6. 手動処理と各ツール

完全自動で作った音声に違和感がある時は、以下のように工程を分けて確認します。音が崩れているのか、Whisper が崩れているのか、LLM 応答が長すぎるのか、TTS が合っていないのかを切り分けやすくなります。

ツール早見表:

| ツール | 使う場面 |
|---|---|
| `processRawToStereo.py` | raw フォルダを一括で stereo 化したい |
| `generateSoloConversationDataset.py` | 1本だけ一人配信を stereo 化して細かく確認したい |
| `trimSilence.py` | 長い無音だけを削りたい |
| `extractVocalsDemucs.py` | サーバ上で簡易的に声を抽出したい |
| `synthesizeTts.py` | TTS API だけ単体で試したい |
| `makeStereoPair.py` | 既にある左右音声を stereo wav にしたい |
| `prepareDatasetJsonl.py` | `datasets/stereo/` から `train.jsonl` を作りたい |
| `annotateDataset.py` | ローカル Whisper で学習用 transcript を作りたい |

### 手動で1本ずつ作る

#### 1. 元音声を `datasets/raw/` に置く

```text
datasets/raw/source001.wav
```

まずは 1 本だけ置いて、音量、ノイズ、BGM、不要な無音を耳で確認します。PC 側で UVR5 を使って声だけにした音声がある場合は、その出力をここへ入れます。

#### 2. 必要なら無音を詰める

```bash
uv run --project moshi-finetune python scripts/trimSilence.py \
  --input datasets/raw/source001.wav \
  --output datasets/raw/trimmed/source001.wav \
  --threshold-db -45 \
  --min-silence-sec 0.8 \
  --keep-silence-sec 0.25
```

無音を詰めすぎると会話の自然な間が消えます。まずは既定値に近い設定で出力を聞き、待機時間だけが削れているか確認してください。

#### 3. 一人配信から1本だけ stereo を作る

左chに元音声、右chに LLM + TTS で作った応答を入れる場合は、単体変換スクリプトを使います。

```bash
uv run --project moshi-finetune python scripts/generateSoloConversationDataset.py \
  --input datasets/raw/trimmed/source001.wav \
  --output datasets/stereo/source001.wav \
  --metadata-output datasets/stereo/source001.responses.json \
  --max-segments 10 \
  --tts-speed 1.2
```

確認ポイント:

| 見る場所 | 確認すること |
|---|---|
| `datasets/stereo/source001.wav` | 左右の音が自然に並んでいるか |
| `datasets/stereo/source001.responses.json` | どの発話にどんな応答が付いたか |
| `datasets/cache/source001/transcript.json` | Whisper の文字起こしが大きく崩れていないか |
| `datasets/cache/source001/responses.json` | LLM 応答の長さや内容が合っているか |
| `datasets/cache/source001/tts/` | 生成された TTS 音声が破綻していないか |

`--max-segments 10` は試作用です。問題なければ外して同じコマンドを再実行します。Whisper 結果は `datasets/cache/` に残るため、通常は最初からやり直しにはなりません。

#### 4. TTS だけ単体で確認する

TTS API の音色や速度を先に確認したい場合:

```bash
uv run --project moshi-finetune python scripts/synthesizeTts.py \
  "そうなんですね、もう少し聞かせてください" \
  --type 10 \
  --output datasets/tts/check001.wav
```

この単体スクリプトは速度変更をしません。学習用の一人配信補完では `generateSoloConversationDataset.py` と `processRawToStereo.py` の `--tts-speed` で、ピッチを変えずに速度調整します。

#### 5. 左右の音声が既にある場合は手動で stereo 化する

コラボ、通話、手作業で作った応答音声など、左chと右chを別ファイルで用意できている場合は、LLM/TTS 補完を使わずに合成できます。

```bash
uv run --project moshi-finetune python scripts/makeStereoPair.py \
  --left datasets/raw/leftSpeaker.wav \
  --right datasets/raw/rightSpeaker.wav \
  --output datasets/stereo/manual001.wav
```

`makeStereoPair.py` は左右を 24kHz にそろえ、短い方を無音で埋めて同じ長さにします。左右の開始タイミングがずれている場合は、事前に DAW や音声編集ソフトで合わせてから使ってください。

#### 6. 良いものだけ JSONL に入れる

単体確認で問題ない wav だけを `datasets/stereo/` に残してから、JSONL を作ります。

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl
```

学習用 transcript まで作ったあとに欠けがないか確認する場合:

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl \
  --require-transcript
```

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

## 7. JSONL と学習用 transcript

`datasets/stereo/*.wav` から `datasets/train.jsonl` を作ります。

```bash
uv run --project moshi-finetune python scripts/prepareDatasetJsonl.py \
  --audio-dir datasets/stereo \
  --output datasets/train.jsonl
```

形式:

```json
{"path": "stereo/sample001.wav", "duration": 12.34}
```

`path` は `datasets/train.jsonl` からの相対パスで保存します。`datasets/stereo/...` と書くと、学習時に `datasets/datasets/stereo/...` と二重解決されて音声が見つからなくなります。

学習用 transcript を生成します。Whisper はローカル実行で、基本は `large-v3` を使います。

```bash
uv run --project moshi-finetune python scripts/annotateDataset.py \
  datasets/train.jsonl \
  --lang ja \
  --whisper-model large-v3 \
  --whisper-cache-dir models/whisper
```

既に近似生成した `datasets/stereo/*.json` がある場合、`annotateDataset.py` は既存ファイルをスキップします。Whisper で作り直す場合は `--overwrite-existing` を付けます。

```bash
uv run --project moshi-finetune python scripts/annotateDataset.py \
  datasets/train.jsonl \
  --lang ja \
  --whisper-model large-v3 \
  --whisper-cache-dir models/whisper \
  --overwrite-existing
```

すでに `processRawToStereo.py` で作った `datasets/stereo/*.responses.json` があり、Whisper を再実行せずにまず学習を通したい場合は、近似 transcript を作れます。

```bash
uv run --project moshi-finetune python scripts/createAnnotationJsonFromResponses.py \
  datasets/train.jsonl
```

これは `*.responses.json` 内のセグメント時刻から `datasets/stereo/*.json` を作る高速な方法です。ただし word timestamp ではなく近似なので、品質優先では `annotateDataset.py` を使ってください。

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

通常は `train-run.sh` を使います。先に `datasets/stereo/*.json` が揃っているか確認し、不足している場合はモデルを読み込む前に停止します。既に `run_dir` の出力先がある場合は、削除せず `*.previous.YYYYMMDD-HHMMSS` に退避してから新しく開始します。

1 GPU:

```bash
bash train-run.sh
```

複数 GPU:

```bash
CUDA_DEVICES=0,1,2,3 bash train-run.sh
```

`train-run.sh` は `CUDA_DEVICES` の数から `torchrun --nproc-per-node` を自動設定します。明示したい場合は `NPROC_PER_NODE` を指定します。内部では PyTorch 標準の `CUDA_VISIBLE_DEVICES` に変換して実行します。

```bash
CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash train-run.sh
```

手動で実行する場合:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}" \
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune torchrun \
  --nproc-per-node 4 \
  --master_port 29501 \
  moshi-finetune/train.py \
  config/llmJpMoshiLora.yaml
```

別の分散ジョブと同時に動かす場合は、`MASTER_PORT=29502 bash train-run.sh` のようにポートを変えてください。

Tesla P40 など CUDA Capability 7.0 未満のGPUでは、PyTorch Inductor / Triton のコンパイルが使えません。`NO_TORCH_COMPILE=1` はその最適化を切って eager 実行にするための設定です。新しいGPUで速度を優先したい場合だけ `NO_TORCH_COMPILE=0` を明示してください。

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

一番簡単な起動:

```bash
bash start.sh
```

`start.sh` は最新 checkpoint を `loras/llmJpMoshiV1/latest/` にコピーしてから、その LoRA で Moshi server を起動します。起動後、`http://localhost:8998` にアクセスします。

推論は `moshi.server` が1プロセス1GPUの構成です。モデルを複数GPUへ分割するのではなく、複数GPUで試したい場合はGPUごとに別ポートでサーバーを起動します。

```bash
CUDA_DEVICES=0 bash start.sh --port 8998
CUDA_DEVICES=1 bash start.sh --port 8999
```

古いGPUで float16 推論にしたい場合:

```bash
bash start.sh --half
```

GPU メモリが足りない場合は、float16 と LoRA 融合の無効化を試します。

```bash
bash start.sh --half --no_fuse_lora
```

`--half` は bfloat16 ではなく float16 で読み込む設定で、特に Tesla P40 など古いGPUではほぼ必須です。`--no_fuse_lora` は LoRA をベース重みに融合する処理を避けるため、起動時の一時的なメモリ増加を抑えられることがあります。

途中の checkpoint を直接テストしたい場合:

```bash
LORA_WEIGHT=loras/llmJpMoshiV1/checkpoints/checkpoint_001200/consolidated/lora.safetensors \
CONFIG_PATH=loras/llmJpMoshiV1/checkpoints/checkpoint_001200/consolidated/config.json \
bash start.sh
```

`LORA_WEIGHT` と `CONFIG_PATH` は同じ checkpoint のペアを指定してください。

手動で起動する場合:

```bash
HF_HOME="$PWD/models/huggingface" \
NO_TORCH_COMPILE=1 \
uv run --project moshi-finetune python -m moshi.server \
  --hf-repo llm-jp/llm-jp-moshi-v1 \
  --lora-weight loras/llmJpMoshiV1/latest/lora.safetensors \
  --config-path loras/llmJpMoshiV1/latest/config.json
```

エコーを避けるため、確認時はイヤホンかヘッドホンを使ってください。

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
