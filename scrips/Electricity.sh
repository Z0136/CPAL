#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"

"$python_bin" -B "$project_root/run.py" "$@" --dataset Electricity --pred_len 96 --run
"$python_bin" -B "$project_root/run.py" "$@" --dataset Electricity --pred_len 192 --run
"$python_bin" -B "$project_root/run.py" "$@" --dataset Electricity --pred_len 336 --run
"$python_bin" -B "$project_root/run.py" "$@" --dataset Electricity --pred_len 720 --run
