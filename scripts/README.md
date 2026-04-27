# scripts 構成

今後の追加先を迷いにくくするため、`scripts/` は役割ごとに分けています。

| パス | 用途 |
| --- | --- |
| `scripts/common/` | Python スクリプト間で共有するユーティリティ |
| `scripts/dataset/` | 一人配信や汎用データセット前処理 |
| `scripts/collab/` | 今後追加するコラボ配信向け前処理 |
| `scripts/train/` | 学習前後の補助処理 |
| `scripts/model/` | モデル取得やキャッシュ準備 |
| `scripts/env/` | shell から使う環境初期化と `.env` ローダー |

Python は `python -m scripts.<group>.<module>` で実行します。

例:

```bash
uv run --project moshi-finetune python -m scripts.dataset.processRawToStereo --dry-run
uv run --project moshi-finetune python -m scripts.train.renderTrainConfig --help
```
