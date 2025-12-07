#!/usr/bin/env bash
set -euo pipefail
python -m ArcticRoute.api.cli paper.build --profile full
python -m ArcticRoute.api.cli paper.video --profile full
python -m ArcticRoute.api.cli paper.bundle --profile full --tag v0-full
python -m ArcticRoute.api.cli paper.check --bundle outputs/release/arcticroute_repro_v0-full.zip






