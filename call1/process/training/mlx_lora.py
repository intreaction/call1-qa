"""``python -m call1.process.training.mlx_lora <mlx_lm lora arguments>``: mlx_lm's LoRA trainer with
the chat template rendered as production renders it (docs/OnDeviceTraining.md section 7.4, L2).

mlx_lm's ``TokenizerWrapper.apply_chat_template`` defaults ``enable_thinking`` to the tokenizer's
``has_thinking``, which is true for Gemma 4, so its chat dataset renders every training prompt with
``<|think|>`` at the top of the system turn. Production (``call1.adapters.mlx.generate_loaded``)
renders with ``enable_thinking=False``, which has no ``<|think|>``. Qualification L2 found the two
differ, so this entry point makes thinking-off the default for the whole training process (both the
full conversation and the ``--mask-prompt`` offset), then runs ``mlx_lm lora`` unchanged. An explicit
``enable_thinking`` argument still wins. Apple Silicon only.
"""

from __future__ import annotations

import sys


def patch_thinking_off() -> None:
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    original = TokenizerWrapper.apply_chat_template
    if getattr(original, "_call1_thinking_off", False):
        return

    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(self, *args, **kwargs)

    apply_chat_template._call1_thinking_off = True  # type: ignore[attr-defined]
    TokenizerWrapper.apply_chat_template = apply_chat_template


def main(argv=None) -> int:  # pragma: no cover - needs MLX and the model weights (local qualification L1)
    patch_thinking_off()
    from mlx_lm import lora

    sys.argv = ["mlx_lm.lora", *(sys.argv[1:] if argv is None else argv)]
    lora.main()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
