#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$ALOHA_LAB_ROOT"
run_id="$(date +%Y%m%d-%H%M%S)"
aloha-lab collect --output "data/smoke-$run_id" --episodes 8 --steps 180
aloha-lab train --data "data/smoke-$run_id" --output "runs/bc-$run_id" --epochs 80
aloha-lab evaluate --checkpoint "runs/bc-$run_id/policy.pt" --output "runs/eval-$run_id"
printf 'Results: %s/runs/eval-%s\n' "$ALOHA_LAB_ROOT" "$run_id"
