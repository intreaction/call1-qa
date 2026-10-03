"""Fresh real-processing demo ingests of the attributed AppTek excerpt.

A WAV INFO comment gives each take a distinct container checksum while preserving every
PCM sample. This avoids ingest deduplication returning a previously processed call.
"""
import io
import struct
import uuid
from pathlib import Path

SAMPLE = Path(__file__).parent / "static" / "demo" / "apptek-retail-short.wav"


def fresh_sample() -> io.BytesIO:
    data = SAMPLE.read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Demo sample is not a WAV recording")
    comment = f"Call1 demo take {uuid.uuid4()}".encode() + b"\0"
    info = b"ICMT" + struct.pack("<I", len(comment)) + comment + (b"\0" if len(comment) % 2 else b"")
    payload = b"INFO" + info
    chunk = b"LIST" + struct.pack("<I", len(payload)) + payload
    out = data + chunk
    return io.BytesIO(out[:4] + struct.pack("<I", len(out) - 8) + out[8:])
