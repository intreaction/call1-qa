# Call1

Local call-quality analysis with transcript evidence, configurable rubrics and human review.

- **Process** imports recordings and runs transcription, speaker attribution, masking, analysis and scoring.
- **Store** owns recordings, results, jobs, reviewer accounts and audit history.
- **Evaluate** is the browser workspace for reviewing calls, coaching and metrics.

The UI runs in your browser. Processing can run on your own Apple Silicon Mac using local models. There is no native desktop shell or installer in this source distribution.

## Run the demo

Use Python 3.12 or newer and FFmpeg on your PATH. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m call1.launch --demo --handlers fake
```

The launcher prints and opens Evaluate, the Store console and the credential-bearing Process console URL. Keep that Process URL private. Demo sign-in and 560 fictional historical sessions are available locally. Fake handlers produce scripted results; they do not measure model accuracy. Peer-center comparisons are illustrative.

The committed frontend builds let you start without Node.js. For real inference, install the MLX requirements and provision the models described in [Operations](docs/Operations.md). The Process demo studio includes an attributed AppTek role-play excerpt for an actual processing run when models are configured.

## Develop and verify

See [Development](docs/Development.md) for frontend builds, generated test audio and test commands. See [Architecture](docs/Architecture.md) for component boundaries and [Operations](docs/Operations.md) for startup, credentials and local data.

## Scope and attribution

This source snapshot includes the current three-app product, shared inference code, relevant tests and required assets/notices. Native macOS packaging, the old combined server/UI, private course submissions, business plans, research dumps and model weights are excluded. Commercial hosting, subscriptions and cross-organization benchmarking remain proposed features.

Built by John Wheeler, Ryan Wolff and Cameron Anthony for the CIS 568 team project. Technical implementation and system design: John; market and competitive analysis: Ryan; business and financial analysis: Cameron.

Core open-source licensing remains pending; this snapshot does not grant an Apache-2.0 license. Preserve [third-party notices](THIRD_PARTY_NOTICES.md) and the component-specific licenses. See [License status](docs/LicenseStatus.md).
