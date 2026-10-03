"""Call audio for reviewer playback, with masking (``mask_reviewer_reads``) and HTTP Range.

The source is the conversation's linked ``source_audio`` artifact (or a linked ``redacted_audio``
artifact when one exists, which is already masked). With masking on, Store mutes the intervals
where sensitive values were spoken (the rule values plus, since contract 1.2.0, the model PII
findings of the current transcript revision; with no findings for it yet, no audio is served:
503 ``pii_findings_pending``), using the pre-split rule
(``call1.redaction.RedactionService.audio_mute_intervals``: word timestamps first, entity times
next, the whole turn when neither aligns). Since team decision 22 it also mutes the spans masked by
position (``masking.call_masks``: read-out window digits and each kept PII finding at its turn and
offsets) and every strong card read-out window whole, from a little before the phrase that opens
it to a little after its last turn (``call1.redaction.read_out_mute_ranges``), so digits the ASR
split, garbled or never transcribed are silent too, in the gaps between turns and on every channel. The muted copy is a content-addressed object cached per
(source, intervals) in ``results_masked_audio`` and served with Range support like any object.

Muting runs in pure Python for PCM WAV. Other containers use ``ffmpeg`` when it is installed. When
muting is impossible Store fails closed (503 ``store_unavailable``, not retryable) and never serves
the unmuted original, as the pre-split app did.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import wave
from types import SimpleNamespace
from typing import List, Optional, Tuple

from starlette.responses import Response

from call1.contracts.artifacts import Artifact, ArtifactKind
from call1.contracts.common import canonical_digest
from call1.contracts.errors import ErrorCode
from call1.redaction import RedactionService

from .. import db
from ..context import Store
from ..db import StoreConnection
from ..errors import StoreError, not_found
from ..queue import api as queue_api
from . import records
from .masking import PII_PATTERNS, call_masks, findings_match, masking_enabled, turn_dicts

_WAV_TYPES = ("audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave")


def mute_intervals(conn: StoreConnection, conversation_id: str) -> List[Tuple[float, float]]:
    """Where to mute: the rule values plus the PII findings of the current transcript revision
    (contract 1.2.0). Raises 503 ``store_unavailable`` (retryable, reason ``pii_findings_pending``)
    while there is no transcript or no findings for it: the audio is never served unmuted."""
    pub, transcript = records.transcript_content(conn, conversation_id)
    findings = records.pii_findings_content(conn, conversation_id) if pub is not None else None
    if pub is None or transcript is None or not findings_match(findings, pub.checksum):
        raise StoreError(ErrorCode.STORE_UNAVAILABLE, "Call audio is withheld until PII masking finishes for this call's transcript",
                         details={"reason": "pii_findings_pending"}, retryable=True)
    return transcript_mute_intervals(turn_dicts(transcript, records.enrichment_content(conn, conversation_id)), findings)


def transcript_mute_intervals(turns: List[dict], findings) -> List[Tuple[float, float]]:
    """The mute intervals for ``turns`` (``masking.turn_dicts``) and their ``findings``: rule values,
    the findings' strong identifiers by value, the positional spans, and the read-out windows."""
    values, positions = call_masks(turns, findings)
    settings = SimpleNamespace(redaction=SimpleNamespace(text=True, audio=True, pii_patterns=PII_PATTERNS))
    return [(float(s), float(e)) for s, e in RedactionService(settings).audio_mute_intervals(  # type: ignore[arg-type]
        [SimpleNamespace(**t) for t in turns], extra_values=values, extra_spans=positions, read_out=True)]


def mute_wav(data: bytes, intervals: List[Tuple[float, float]]) -> bytes:
    """Zero the PCM samples inside ``intervals`` (silence is 0x80 for 8-bit unsigned PCM)."""
    with wave.open(io.BytesIO(data), "rb") as reader:
        params = reader.getparams()
        frames = bytearray(reader.readframes(params.nframes))
    frame_size = params.sampwidth * params.nchannels
    silence = b"\x80" if params.sampwidth == 1 else b"\x00"
    for start, end in intervals:
        first = max(0, int(start * params.framerate))
        last = min(params.nframes, int(end * params.framerate + 0.999999))
        if last > first:
            frames[first * frame_size:last * frame_size] = silence * ((last - first) * frame_size)
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(bytes(frames))
    return out.getvalue()


def _mute_with_ffmpeg(data: bytes, content_type: str, intervals: List[Tuple[float, float]]) -> Optional[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None
    suffix, codec = {"audio/mpeg": (".mp3", ["-c:a", "libmp3lame", "-q:a", "2"]), "audio/mp4": (".m4a", ["-c:a", "aac"]),
                     "audio/flac": (".flac", ["-c:a", "flac"]), "audio/ogg": (".ogg", ["-c:a", "libvorbis"])}.get(content_type, (None, None))
    if suffix is None:
        return None
    with tempfile.TemporaryDirectory(prefix="call1_mute_") as tmp:
        source, target = os.path.join(tmp, "in" + suffix), os.path.join(tmp, "out" + suffix)
        with open(source, "wb") as handle:
            handle.write(data)
        expr = ",".join(f"volume=enable='between(t,{s:.3f},{e:.3f})':volume=0" for s, e in intervals)
        proc = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", source, "-af", expr, *codec, target],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120, check=False)
        if proc.returncode != 0 or not os.path.exists(target) or os.path.getsize(target) == 0:
            return None
        with open(target, "rb") as handle:
            return handle.read()


def _source(conn: StoreConnection, conversation_id: str) -> Tuple[Artifact, bool]:
    redacted = queue_api.list_linked_artifacts(conn, conversation_id, kind=ArtifactKind.REDACTED_AUDIO)
    if redacted:
        return max(redacted, key=lambda a: a.version or 0), True
    sources = queue_api.list_linked_artifacts(conn, conversation_id, kind=ArtifactKind.SOURCE_AUDIO)
    if not sources:
        raise not_found("Call audio", conversation_id=conversation_id)
    return max(sources, key=lambda a: a.version or 0), False


def call_audio(store: Store, conn: StoreConnection, call_id: str) -> Response:
    row = records.call_row(conn, call_id)
    if row is None:
        raise not_found("Call", call_id=call_id)
    with db.read_snapshot(conn):
        artifact, already_masked = _source(conn, row["conversation_id"])
        intervals = [] if already_masked or not masking_enabled() else mute_intervals(conn, row["conversation_id"])
    media_type = artifact.content_type
    if not intervals:
        return store.objects.file_response(artifact.checksum, media_type)
    digest = canonical_digest([[round(s, 3), round(e, 3)] for s, e in intervals])
    cached = conn.execute("SELECT masked_checksum FROM results_masked_audio WHERE source_checksum = ? AND intervals_digest = ?",
                          (artifact.checksum, digest)).fetchone()
    if cached is not None and store.objects.exists(cached["masked_checksum"]):
        return store.objects.file_response(cached["masked_checksum"], media_type)
    data = store.objects.read_bytes(artifact.checksum)
    masked: Optional[bytes] = None
    if media_type.lower() in _WAV_TYPES:
        try:
            masked = mute_wav(data, intervals)
        except (wave.Error, EOFError):
            masked = None
    if masked is None:
        masked = _mute_with_ffmpeg(data, media_type.lower(), intervals)
    if masked is None:
        raise StoreError(ErrorCode.STORE_UNAVAILABLE, "Audio masking is unavailable for this recording; the unmasked audio is not served",
                         details={"reason": "audio_masking_unavailable", "content_type": media_type[:200]}, retryable=False)
    stored = store.objects.put_bytes(masked)
    with db.transaction(conn):
        conn.execute(
            "INSERT OR REPLACE INTO results_masked_audio (source_checksum, intervals_digest, masked_checksum, content_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (artifact.checksum, digest, stored.checksum, media_type, db.ts(conn.now())))
    return store.objects.file_response(stored.checksum, media_type)
