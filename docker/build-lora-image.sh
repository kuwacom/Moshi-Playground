#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# README と同じ .env 解決を使い、LoRA ごとの既定パスを揃える
# shellcheck source=/dev/null
source "$repo_root/scripts/env/loadLoraEnv.sh"

usage() {
  cat <<'EOF'
Usage:
  bash docker/build-lora-image.sh --image IMAGE[:TAG] [options]

Options:
  --image IMAGE[:TAG]        作成する Docker image 名
  --lora-dir DIR             lora.safetensors と config.json が入ったディレクトリ
  --lora-weight FILE         LoRA weight のパス
  --config-path FILE         LoRA と同じ checkpoint の config.json
  --hf-repo REPO             ベースモデルの Hugging Face repo
  --include-hf-cache         models/huggingface をイメージへ同梱する
  --hf-home DIR              同梱する HF_HOME ディレクトリ
  --platform PLATFORM        docker build の --platform
  --moshi-commit COMMIT      使用する kyutai-labs/moshi の commit
  --pytorch-image IMAGE      ベースにする PyTorch CUDA image
  --sudo-docker              sudo docker でビルドする
  -h, --help                 このヘルプを表示する

Examples:
  bash docker/build-lora-image.sh --image moshi-lora:local
  bash docker/build-lora-image.sh --image yourname/moshi-lora:test --lora-dir loras/example/latest
  bash docker/build-lora-image.sh --image ghcr.io/yourname/moshi-lora:test --include-hf-cache
EOF
}

image_ref="${IMAGE_REF:-moshi-lora-${LORA_NAME}:local}"
lora_dir="${LORA_DIR:-$LORA_LATEST_DIR}"
lora_weight="${LORA_WEIGHT:-}"
config_path="${MOSHI_CONFIG_PATH:-${CONFIG_PATH:-}}"
hf_repo="${HF_REPO:-llm-jp/llm-jp-moshi-v1}"
hf_home="${HF_HOME:-$repo_root/models/huggingface}"
include_hf_cache=0
platform="${PLATFORM:-}"
moshi_commit="${MOSHI_COMMIT:-061cc4c630d9e11722e08b7d02b1836ba58f30e8}"
pytorch_image="${PYTORCH_IMAGE:-pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime}"
docker_cmd=(docker)

if [ "${USE_SUDO_DOCKER:-0}" = "1" ]; then
  docker_cmd=(sudo docker)
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --image)
      image_ref="${2:?--image requires a value}"
      shift 2
      ;;
    --lora-dir)
      lora_dir="${2:?--lora-dir requires a value}"
      shift 2
      ;;
    --lora-weight)
      lora_weight="${2:?--lora-weight requires a value}"
      shift 2
      ;;
    --config-path)
      config_path="${2:?--config-path requires a value}"
      shift 2
      ;;
    --hf-repo)
      hf_repo="${2:?--hf-repo requires a value}"
      shift 2
      ;;
    --include-hf-cache)
      include_hf_cache=1
      shift
      ;;
    --hf-home)
      hf_home="${2:?--hf-home requires a value}"
      shift 2
      ;;
    --platform)
      platform="${2:?--platform requires a value}"
      shift 2
      ;;
    --moshi-commit)
      moshi_commit="${2:?--moshi-commit requires a value}"
      shift 2
      ;;
    --pytorch-image)
      pytorch_image="${2:?--pytorch-image requires a value}"
      shift 2
      ;;
    --sudo-docker)
      docker_cmd=(sudo docker)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

resolve_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$repo_root/$1" ;;
  esac
}

lora_dir_abs="$(resolve_path "$lora_dir")"
if [ -z "$lora_weight" ]; then
  lora_weight="$lora_dir_abs/lora.safetensors"
else
  lora_weight="$(resolve_path "$lora_weight")"
fi

if [ -z "$config_path" ]; then
  config_path="$lora_dir_abs/config.json"
else
  config_path="$(resolve_path "$config_path")"
fi

hf_home="$(resolve_path "$hf_home")"

if [ ! -f "$lora_weight" ]; then
  echo "LoRA weight not found: $lora_weight" >&2
  exit 1
fi

if [ ! -f "$config_path" ]; then
  echo "Config not found: $config_path" >&2
  exit 1
fi

if [ "$include_hf_cache" = "1" ] && [ ! -d "$hf_home" ]; then
  echo "HF cache not found: $hf_home" >&2
  exit 1
fi

context_root="$repo_root/docker/.build-contexts"
mkdir -p "$context_root"
build_context="$(mktemp -d "$context_root/moshi-lora.XXXXXX")"

cleanup() {
  rm -rf "$build_context"
}
trap cleanup EXIT

mkdir -p "$build_context/lora" "$build_context/huggingface"
cp "$repo_root/docker/Dockerfile" "$build_context/Dockerfile"
cp "$repo_root/docker/start-lora-server.sh" "$build_context/start-lora-server.sh"
cp "$lora_weight" "$build_context/lora/lora.safetensors"
cp "$config_path" "$build_context/lora/config.json"

if [ "$include_hf_cache" = "1" ]; then
  echo "Copying HF cache into build context: $hf_home"
  cp -a "$hf_home/." "$build_context/huggingface/"
else
  echo "HF cache is not included. The container will download the base model at runtime if needed."
fi

build_args=(
  build
  --tag "$image_ref"
  --build-arg "HF_REPO=$hf_repo"
  --build-arg "LORA_NAME=$LORA_NAME"
  --build-arg "MOSHI_COMMIT=$moshi_commit"
  --build-arg "PYTORCH_IMAGE=$pytorch_image"
)

if [ -n "$platform" ]; then
  build_args+=(--platform "$platform")
fi

build_args+=("$build_context")

echo "Building image: $image_ref"
echo "  lora:    $lora_weight"
echo "  config:  $config_path"
echo "  hf_repo: $hf_repo"

"${docker_cmd[@]}" "${build_args[@]}"

echo "Built image: $image_ref"
