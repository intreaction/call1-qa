"""Explicit build/setup operation. Inference never downloads models.

``asr_vocabulary`` (Whisper Small for dual transcription's vocabulary pass, docs/DualAsr.md) also
bundles openai-whisper's ``multilingual.tiktoken`` into the model directory, from the pinned source in
``model-manifest.json`` or from a local copy (``--tokenizer-from``, e.g. openai-whisper's
``whisper/assets/multilingual.tiktoken`` in the uv cache, no network). Its sha256 must match the pin.
"""
import json
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

KINDS = ['asr', 'asr_vocabulary', 'text', 'diarization', 'tone', 'sentiment', 'privacy_filter']


def install_tokenizer(spec: dict, directory: Path, source: 'Path | None') -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from call1.pipeline.vocabulary_asr import TOKENIZER_SHA256, install_tokenizer as bundle

    tokenizer = spec['tokenizer']
    if tokenizer['sha256'] != TOKENIZER_SHA256:
        raise RuntimeError('model-manifest.json and call1.pipeline.vocabulary_asr pin different tokenizer checksums.')
    bundle(directory, source=source, url=tokenizer['url'])


def install_license(spec: dict, directory: Path) -> None:
    """Writes the pinned licence beside the weights. A copy already in place with the pinned sha256
    is kept as is, with no network access, so an offline ``--tokenizer-only`` run succeeds."""
    license_spec = spec['license']
    target = directory / license_spec.get('filename', 'LICENSE.pdf')
    if not (target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == license_spec['sha256']):
        with urllib.request.urlopen(license_spec['url'], timeout=60) as response:
            license_bytes = response.read()
        if hashlib.sha256(license_bytes).hexdigest() != license_spec['sha256']:
            raise RuntimeError('Model license changed; review the new license before updating its pin.')
        target.write_bytes(license_bytes)
    (directory / 'Notice').write_text(spec['notice'] + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=KINDS, default=list(KINDS),
                        help='Select models to provision; defaults to the complete pipeline.')
    parser.add_argument('--tokenizer-from', type=Path, default=None,
                        help='asr_vocabulary: a local multilingual.tiktoken to bundle instead of downloading it (checksum still verified).')
    parser.add_argument('--tokenizer-only', action='store_true',
                        help='asr_vocabulary: only bundle the tokenizer into an existing model directory (no weights download). '
                             'Offline once the licence is already in place with its pinned checksum.')
    args = parser.parse_args()
    manifest = json.loads(Path('model-manifest.json').read_text())
    for kind in args.models:
        spec = manifest[kind]
        directory = Path('data/models') / spec['directory']
        if not (kind == 'asr_vocabulary' and args.tokenizer_only):
            from huggingface_hub import snapshot_download

            snapshot_download(spec['repository'], revision=spec['revision'], local_dir=directory,
                              ignore_patterns=spec.get('ignore_patterns'), allow_patterns=spec.get('allow_patterns'))
        if kind == 'asr_vocabulary':
            install_tokenizer(spec, directory, args.tokenizer_from)
        if spec.get('notice'):
            (directory / 'Notice').write_text(spec['notice'] + '\n')
        if spec.get('license'):
            install_license(spec, directory)


if __name__ == '__main__':
    main()
