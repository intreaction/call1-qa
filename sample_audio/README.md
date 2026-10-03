# Test and demo data

`manifest.json` describes five synthetic scenarios and their expected durations/channels. Their original macOS System Voices recordings are excluded. Run `python scripts/generate_test_audio.py` for deterministic non-speech PCM fixtures used by fake-handler tests. Generated tones are ignored, clearly marked, and unsuitable for ASR or real-model evaluation.

The AppTek subdirectories retain attributed transcript manifests for regression tests. Full dataset audio and raw metadata dumps are not distributed here. The separately attributed 15.5-second demo excerpt remains at `call1/process/static/demo/`, with its own notice.

All demo people, call history and peer-center benchmarks are illustrative. AppTek source recordings are human role-played scenarios, not real customer calls. Keep dataset attribution and licenses separate from the application source.
