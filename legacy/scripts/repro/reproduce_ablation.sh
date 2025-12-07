#!/usr/bin/env bash
set -euo pipefail
python -m ArcticRoute.api.cli paper.build --profile ablation
python -m ArcticRoute.api.cli paper.bundle --profile ablation --tag v0-ablation
python -m ArcticRoute.api.cli paper.check --bundle outputs/release/arcticroute_repro_v0-ablation.zip






