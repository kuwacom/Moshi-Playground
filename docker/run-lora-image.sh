#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash docker/run-lora-image.sh --image IMAGE[:TAG] [moshi.server options]

Options:
  --image IMAGE[:TAG]  起動する Docker image 名
  --port PORT          ホスト側とコンテナ側のポート
  --gpus VALUE         docker run --gpus の値
  --hf-home DIR        ホスト側の Hugging Face cache を読み取り専用で mount する
  --sudo-docker        sudo docker で起動する
  -h, --help           このヘルプを表示する

Examples:
  bash docker/run-lora-image.sh --image moshi-lora:local
  bash docker/run-lora-image.sh --image moshi-lora:local -- --half
  bash docker/run-lora-image.sh --image moshi-lora:local -- --half --no_fuse_lora
EOF
}

image_ref="${IMAGE_REF:-moshi-lora:local}"
port="${PORT:-8998}"
gpus="${GPUS:-all}"
hf_home="${HF_HOME_MOUNT:-}"
server_args=()
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
    --port)
      port="${2:?--port requires a value}"
      shift 2
      ;;
    --gpus)
      gpus="${2:?--gpus requires a value}"
      shift 2
      ;;
    --hf-home)
      hf_home="${2:?--hf-home requires a value}"
      shift 2
      ;;
    --sudo-docker)
      docker_cmd=(sudo docker)
      shift
      ;;
    --)
      shift
      server_args+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      server_args+=("$1")
      shift
      ;;
  esac
done

run_args=(
  run
  --rm
  -it
  --gpus "$gpus"
  -p "$port:$port"
  -e "PORT=$port"
)

if [ -n "$hf_home" ]; then
  case "$hf_home" in
    /*) ;;
    *) hf_home="$PWD/$hf_home" ;;
  esac
  run_args+=(
    -v "$hf_home:/opt/moshi/models/huggingface:ro"
    -e "HF_HOME=/opt/moshi/models/huggingface"
  )
fi

run_args+=("$image_ref")

"${docker_cmd[@]}" "${run_args[@]}" "${server_args[@]}"
