# Third-party components and optional models

This source snapshot includes code, web assets and the attributed demo excerpt.
Model weights and system binaries are installed separately. Preserve the relevant
upstream licenses and notices; model entries below describe optional runtime components.

- SeaweedFS 4.46: Apache-2.0, https://github.com/seaweedfs/seaweedfs/tree/4.46
- Outlines and Outlines Core: Apache-2.0, https://github.com/dottxt-ai/outlines
- MLX and MLX LM: MIT, https://github.com/ml-explore
- MLX Audio 0.5.6: MIT, https://github.com/Blaizzy/mlx-audio (runs Parakeet,
  Nemotron-3-Diarization and the Whisper Small vocabulary pass on Apple Silicon)
- Gemma 4 E2B (4-bit MLX conversion): Apache-2.0;
  https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit
- Parakeet TDT 0.6B v3 (ASR on Apple Silicon): NVIDIA `nvidia/parakeet-tdt-0.6b-v3`,
  MLX conversion `mlx-community/parakeet-tdt-0.6b-v3` at the revision in
  `model-manifest.json`. Licensed under the Creative Commons Attribution 4.0
  International license (CC-BY-4.0), https://creativecommons.org/licenses/by/4.0/.
  Attribution: "Parakeet TDT 0.6B v3 by NVIDIA, MLX conversion by MLX Community,
  CC-BY-4.0." Modification notice: Call1 casts the weights to bfloat16 when loading
  them; the distributed files are unmodified. No warranty is given (CC-BY-4.0 section 5).
  The license text (`LICENSE.CC-BY-4.0.txt`), the attribution `Notice` and the model
  card accompany the weights under `models/parakeet-tdt-0.6b-v3`.
- Whisper Small (vocabulary pass of dual transcription on Apple Silicon, decision 33,
  `docs/DualAsr.md`): OpenAI Whisper, MIT, Copyright (c) 2022 OpenAI,
  https://github.com/openai/whisper. MLX conversion `mlx-community/whisper-small-mlx` at the
  revision in `model-manifest.json`, run through MLX Audio. Used only to find vocabulary
  candidates, never as the transcript. The MIT license text (`LICENSE.whisper-MIT.txt`) accompanies
  the weights under `models/whisper-small`, with the tokenizer vocabulary `multilingual.tiktoken`
  bundled from openai/whisper v20250625 (MIT). openai-whisper's `tokenizer.py` is vendored as
  `call1/pipeline/_vendor/whisper_tokenizer.py` with its MIT notice in the file header; it needs
  `tiktoken` 0.14.0 (MIT, https://github.com/openai/tiktoken). The weights are not modified.
- Whisper small on the non-Mac CPU backend: faster-whisper, MIT,
  https://github.com/SYSTRAN/faster-whisper
- FFmpeg and its linked codec libraries: build-specific LGPL/GPL notices;
  https://ffmpeg.org/legal.html. FFmpeg is a separately installed runtime dependency;
  no FFmpeg binaries or linked libraries are redistributed in this source snapshot.
- Python, FastAPI, SQLite and other Python packages retain their bundled notices.
- py_webauthn (PyPI package `webauthn`, >=3,<4): BSD-3-Clause,
  https://github.com/duo-labs/py_webauthn. Server-side passkey registration and
  authentication for Store's reviewer sign-in.
- @simplewebauthn/browser (^13.3.0): MIT, https://github.com/MasterKale/SimpleWebAuthn.
  Browser-side passkey enrollment and sign-in for Evaluate.
- @playwright/test (^1.63.0): Apache-2.0, https://github.com/microsoft/playwright. Dev/test
  dependency only, used for end-to-end tests; not bundled in any shipped build.

Public replay data is from [AppTek Call-Center Dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues),
© AppTek, under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
These are human role-played scenarios. Source recordings/transcripts and Call1's
reformatted dataset annotations retain that separate license. See the READMEs in
`sample_audio/apptek_retail/` and `sample_audio/telephony_training_set/` for provenance.
The Process web bundle also includes a 15.5-second adapted excerpt; its exact source,
revision, edits and license are in `call1/process/static/demo/NOTICE.md`. This corrects
the earlier statement that no dataset audio was included in an application bundle.
The upstream card specifies evaluation/analysis as the intended use, not model training.
Model revisions and the storage binary checksum are recorded in model-manifest.json.
# Mono speaker clustering

MLX Audio 0.5.6 is MIT licensed. The bundled diarization weights are NVIDIA's
`nvidia/Nemotron-3-Diarization` (up to eight speakers), converted by MLX Community
(`mlx-community/Nemotron-3-Diarization`) at the revision recorded in
`model-manifest.json`. They use the OpenMDW License Agreement, version 1.1
(OpenMDW-1.1), not the Python library's MIT license. OpenMDW-1.1 requires any
distribution of the model materials to retain (1) a copy of the agreement and
(2) all copyright notices and other notices of origin included in the materials;
it places no restrictions on outputs, and rights terminate for anyone who sues
alleging the materials infringe a patent or copyright. The license text
(`LICENSE.OpenMDW-1.1`), the origin `Notice` and the model card (`README.md`)
accompany the weights under `models/nemotron-3-diarization`.
Miniaudio 1.71 and sounddevice 0.5.6 are pinned audio dependencies.

## Search embeddings

Process and Store both run NVIDIA's `nvidia/Nemotron-3-Embed-1B-BF16` (decision 18) through
PyTorch and Transformers, at the revision recorded in `model-manifest.json`. The weights are
licensed under the OpenMDW License Agreement, version 1.1 (Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES), with the same obligations as the diarization weights above: any distribution retains
the agreement and all copyright and origin notices. The model is derived from
`mistralai/Ministral-3-3B-Instruct-2512` (Apache-2.0), and the repository's `NOTICE` carries that
Apache-2.0 attribution. The repository's `LICENSE`, `NOTICE` and `THIRD_PARTY_NOTICES.md` accompany
the weights under `models/nemotron-3-embed-1b`. The weights are not modified.

## PII masking model

Process's masking step runs OpenAI's `openai/privacy-filter` (decision 19) through PyTorch and
Transformers (`OpenAIPrivacyFilterForTokenClassification`, no remote code), at the revision
recorded in `model-manifest.json`. The weights, tokenizer, config and `viterbi_calibration.json`
are licensed under the Apache License, Version 2.0; only the repository's root files are
distributed (not `onnx/` or `original/`). The Apache-2.0 license text (`LICENSE.txt`), the origin
`Notice` and the model card (`README.md`) accompany the weights under
`models/openai-privacy-filter`. The weights are not modified.

## Sentiment models

- Selected VAD model: A*STAR `MERaLiON/MERaLiON-SER-v1`.
  https://huggingface.co/MERaLiON/MERaLiON-SER-v1
  Copyright 2025 AGENCY FOR SCIENCE TECHNOLOGY AND RESEARCH.
  The MERaLiON Public Licence includes an MIT commercial-use grant plus additional
  attribution, modification-notice, redistribution, and derivative-work terms.
  Preserve the full `MERaLiON-Public-Licence-SER-V1.pdf` accompanying the model.
  Speech-only valence, arousal, and dominance are output in [0,1]. This model uses
  a Whisper backbone. The checkpoint is 3.08 GB, including an unused decoder;
  the model card describes 309M parameters in the inference architecture.
- SpeechBrain `emotion-diarization-wavlm-large`: published under Apache-2.0.
  https://huggingface.co/speechbrain/emotion-diarization-wavlm-large
  The pinned checkpoint and Apache license are in `data/research-models/wavlm-emotion-speechbrain`.
  Retained comparison model: outputs angry, neutral, happy, and sad intervals,
  not dimensional valence/arousal. Superseded by MERaLiON for the VAD requirement.
- Cardiff NLP `twitter-roberta-base-sentiment-latest`: CC-BY-4.0, TimeLMs / TweetEval.
  https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
  The pinned checkpoint and original model card are in `data/models/roberta-sentiment`.
  English text sentiment; call-domain quality is not yet validated.

Download these with `scripts/provision_models.py --models tone sentiment`.
This selects MERaLiON VAD and Cardiff NLP text sentiment, now connected to pipeline
inference. The adapted MERaLiON architecture in `call1/adapters/meralion` retains
its license and modification notice. It loads only the encoder and prediction
heads, with no remote code execution or base-model downloads. The earlier
Vox-Profile MSP-Podcast checkpoint prohibits commercial use and is retained only
under `data/research-models/wavlm-emotion-noncommercial`, outside the packaged
`data/models` tree.

## Evaluate chart components (2026-10-02)

Evaluate uses Recharts and React Is (versions pinned in `frontend/package-lock.json`) and adapts
shadcn/ui's MIT chart component to Call1 theme tokens. The shadcn copyright and permission text
is retained in `frontend/src/apps/evaluate/components/charts/shadcn-license.txt`.

### Recharts

The MIT License (MIT)

Copyright (c) 2015-present recharts

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### React Is

MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### shadcn/ui

MIT License

Copyright (c) 2023 shadcn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
