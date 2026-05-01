#!/usr/bin/env python3
from __future__ import annotations

import json

from algae_model.config import default_config
from algae_model.train import train_pipeline


def main() -> None:
    manifest = train_pipeline(default_config())
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
