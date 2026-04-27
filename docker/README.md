# Moshi LoRA Docker 推論

学習後の `lora.safetensors` と `config.json` を Docker image に入れて、GPU コンテナ環境で `moshi.server` を起動するための最小構成です。

通常は repo root の `start.sh` と同じく、以下のペアを使います。

```text
loras/${LORA_NAME}/latest/lora.safetensors
loras/${LORA_NAME}/latest/config.json
```

## 1. ローカルで image を作る

repo root から実行します。

```bash
bash docker/build-lora-image.sh --image moshi-lora:local
```

別の LoRA を明示する場合:

```bash
bash docker/build-lora-image.sh \
  --image moshi-lora:local \
  --lora-dir loras/example-lora/latest
```

checkpoint を直接指定する場合:

```bash
bash docker/build-lora-image.sh \
  --image moshi-lora:checkpoint-1200 \
  --lora-weight loras/example-lora/checkpoints/checkpoint_001200/consolidated/lora.safetensors \
  --config-path loras/example-lora/checkpoints/checkpoint_001200/consolidated/config.json
```

## 2. ベースモデルを image に入れるか選ぶ

既定では LoRA だけを image に入れます。ベースモデル `llm-jp/llm-jp-moshi-v1` はコンテナ初回起動時に Hugging Face から取得されます。

GPU コンテナサービス側で外部ネットワークを使えない、または起動時間を短くしたい場合は、ローカルの `models/huggingface` も同梱します。

```bash
bash docker/build-lora-image.sh \
  --image moshi-lora:with-model \
  --include-hf-cache
```

この場合、image は 15GB 以上大きくなります。事前にモデルを取得していない場合は、repo root で以下を実行します。

```bash
HF_HOME="$PWD/models/huggingface" \
uv run --project moshi-finetune python -m scripts.model.downloadModel
```

## 3. ローカルで起動確認する

```bash
bash docker/run-lora-image.sh --image moshi-lora:local
```

起動後、ブラウザで `http://localhost:8998` を開きます。

ローカルの Hugging Face cache を使って起動時ダウンロードを避ける場合:

```bash
bash docker/run-lora-image.sh \
  --image moshi-lora:local \
  --hf-home models/huggingface
```

古い GPU で float16 を使う場合:

```bash
bash docker/run-lora-image.sh --image moshi-lora:local -- --half
```

GPU メモリが足りない場合:

```bash
bash docker/run-lora-image.sh --image moshi-lora:local -- --half --no_fuse_lora
```

`docker` の権限がない環境では、補助スクリプトに `--sudo-docker` を付けます。

```bash
bash docker/build-lora-image.sh --image moshi-lora:local --sudo-docker
bash docker/run-lora-image.sh --image moshi-lora:local --sudo-docker
```

## 4. Docker Hub に push する

```bash
docker login

bash docker/build-lora-image.sh \
  --image your-dockerhub-name/moshi-lora:test \
  --include-hf-cache

docker push your-dockerhub-name/moshi-lora:test
```

コンテナサービス側では、公開または pull 可能な private image として `your-dockerhub-name/moshi-lora:test` を指定します。テスト後に消す場合は Docker Hub の repository 画面から tag または repository を削除し、ローカルも不要なら以下で消します。

```bash
docker image rm your-dockerhub-name/moshi-lora:test
```

## 5. GitHub Container Registry に push する

GitHub の Personal access token には `write:packages` を付けます。

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u your-github-name --password-stdin

bash docker/build-lora-image.sh \
  --image ghcr.io/your-github-name/moshi-lora:test \
  --include-hf-cache

docker push ghcr.io/your-github-name/moshi-lora:test
```

private package の場合、GPU コンテナサービス側で GHCR の認証情報を設定してください。テスト後に消す場合は GitHub の Packages 画面から package または version を削除し、ローカルも不要なら以下で消します。

```bash
docker image rm ghcr.io/your-github-name/moshi-lora:test
```

## 6. よく変える値

| 目的 | 指定 |
| ---- | ---- |
| image 名を変える | `--image your/name:tag` |
| LoRA latest を変える | `--lora-dir loras/name/latest` |
| checkpoint を直接試す | `--lora-weight ... --config-path ...` |
| HF repo を変える | `--hf-repo owner/repo` |
| HF cache を同梱する | `--include-hf-cache` |
| amd64 を明示する | `--platform linux/amd64` |

コンテナ内の既定値:

| 値 | 既定 |
| ---- | ---- |
| `HF_HOME` | `/opt/moshi/models/huggingface` |
| `LORA_WEIGHT` | `/opt/moshi/lora/lora.safetensors` |
| `MOSHI_CONFIG_PATH` | `/opt/moshi/lora/config.json` |
| `HOST` | `0.0.0.0` |
| `PORT` | `8998` |
| `NO_TORCH_COMPILE` | `1` |
