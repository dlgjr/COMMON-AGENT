#!/usr/bin/env bash
set -Eeuo pipefail

export OPD_STAGE=1
exec bash /mnt/nas/bihaoran/common_agent/scripts/opd/common/run_opd_stage.sh "$@"
