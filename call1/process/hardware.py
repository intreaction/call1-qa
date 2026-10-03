"""The measured hardware profile of this Process host (``PUT /hardware-profiles``).

Usage rows name the profile they ran on. The fingerprint is ``canonical_digest`` of the fields, so
Store upserts one profile per distinct host and runtime set. Runtime versions are read from package
metadata, never by importing a runtime (importing MLX or torch here would load them in every mode).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Dict

from call1.contracts.usage import HardwareProfileFields, HardwareProfileInput, HardwareProfileKind, HardwareProfileSource, hardware_fingerprint

RUNTIME_PACKAGES = ("mlx", "mlx-lm", "mlx-audio", "torch", "transformers")


def _chip() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()[:200]
        except (OSError, subprocess.SubprocessError):
            pass
    return (platform.processor() or platform.machine() or "unknown")[:200]


def _memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def runtime_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def measure() -> HardwareProfileFields:
    return HardwareProfileFields(
        kind=HardwareProfileKind.PROCESS_HOST, source=HardwareProfileSource.MEASURED, chip=_chip(), accelerator=None,
        memory_bytes=_memory_bytes(), os_name=platform.system() or "unknown", os_version=(platform.release() or "unknown")[:200],
        runtime_versions=runtime_versions(),
    )


def profile_input(fields: HardwareProfileFields | None = None) -> HardwareProfileInput:
    fields = fields or measure()
    return HardwareProfileInput(**fields.model_dump(), fingerprint=hardware_fingerprint(fields))
