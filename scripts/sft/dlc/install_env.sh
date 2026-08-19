#!/usr/bin/env bash
set -euo pipefail

PY="/mnt/nas/bihaoran/common_agent/envs/qwen35_swift/bin/python"

"$PY" -m pip install --ignore-installed --no-deps "peft==0.18.0"
"$PY" -m pip install "transformers==5.2.0" "datasets>=3.0,<4.8.5" "pyarrow>=18.0.0" tensorboard
