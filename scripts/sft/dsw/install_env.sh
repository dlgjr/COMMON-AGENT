#!/usr/bin/env bash
PY="/mnt/nas/bihaoran/common_agent/envs/qwen35_swift/bin/python"
"$PY" -m pip install "transformers==5.2.0" "datasets>=4.0.0" "pyarrow>=18.0.0" tensorboard
