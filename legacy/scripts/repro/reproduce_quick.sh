#!/usr/bin/env bash
set -euo pipefail
python -m ArcticRoute.api.cli paper.build --profile quick
python -m ArcticRoute.api.cli paper.video --profile quick
python -m ArcticRoute.api.cli paper.bundle --profile quick --tag v0-quick
python -m ArcticRoute.api.cli paper.check --bundle outputs/release/arcticroute_repro_v0-quick.zip






