"""Serialize local inference so ASR and summaries do not compete for VRAM."""

from threading import RLock

inference_lock = RLock()
