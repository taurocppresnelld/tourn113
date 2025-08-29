#!/bin/bash
set -e
echo "[run_text_trainer.sh] Starting trainer with args: $@"
python3 /workspace/scripts/text_trainer.py "$@"