"""The fake trainer process: ``python -m call1.process.training.fake_trainer --data D --adapter-path A
--iters N --seed S``.

It reads the training files (so a malformed dataset fails here as it would in mlx_lm), prints
mlx_lm-shaped progress lines and writes a small ``adapters.safetensors`` and
``adapter_config.json``. It never imports MLX or torch.

* ``CALL1_FAKE_TRAINER_SECONDS``: total seconds to spend (spread over the iterations), for slow,
  cancelled and timed-out runs.
* ``CALL1_FAKE_TRAINER_EXIT``: exit with this code after printing, without writing an adapter (a
  crash or an out-of-memory kill).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="call1.process.training.fake_trainer")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    data = Path(args.data)
    train = [json.loads(line) for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in train:
        roles = [m.get("role") for m in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            print("fake trainer: a training row is not system, user, assistant", flush=True)
            return 2
    seconds = float(os.getenv("CALL1_FAKE_TRAINER_SECONDS") or 0)
    iters = max(1, args.iters)
    reports = min(iters, 10)
    step = max(1, iters // reports)
    pause = seconds / reports if reports else 0.0
    print(f"Loading pretrained model (fake); {len(train)} training examples", flush=True)
    print("Iter 1: Val loss 2.500, Val took 0.100s", flush=True)
    loss = 2.4
    for i in range(step, iters + 1, step):
        if pause:
            time.sleep(pause)
        loss = max(0.1, loss * 0.85)
        print(f"Iter {i}: Train loss {loss:.3f}, Learning Rate 1.000e-04, It/sec 2.500, Tokens/sec 900.0, Trained Tokens {i * 800}, "
              f"Peak mem 1.250 GB", flush=True)
    print(f"Iter {iters}: Val loss {loss * 1.1:.3f}, Val took 0.100s", flush=True)
    code = os.getenv("CALL1_FAKE_TRAINER_EXIT")
    if code and code != "0":
        print("fake trainer: exiting as asked", flush=True)
        return int(code)
    out = Path(args.adapter_path)
    out.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"__metadata__": {"format": "fake", "seed": str(args.seed)}}).encode("utf-8")
    (out / "adapters.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    digest = hashlib.sha256((data / "train.jsonl").read_bytes()).hexdigest()
    (out / "adapter_config.json").write_text(json.dumps({"fine_tune_type": "lora", "num_layers": 16, "fake": True, "data_digest": digest,
                                                         "iters": iters, "seed": args.seed}, indent=2))
    print(f"Saved final weights to {out / 'adapters.safetensors'}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
