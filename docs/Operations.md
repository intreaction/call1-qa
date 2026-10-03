# Operations

## Local demo

Run `python -m call1.launch --demo --handlers fake` from the repository root. Evaluate and Store use `http://localhost:8010`; Process uses `http://127.0.0.1:8020`. Open the Process URL printed at startup, which includes its console credential. Stop both services with Ctrl-C. Add `--no-open` to suppress browser launch.

Demo mode uses `data/demo/`, persona sign-in and fictional seeded history. `--reset` clears only the selected demo workspace after the launcher's safety checks; use it when you intend to restart the demo. Keep demo mode on localhost.

## Real inference

On Apple Silicon, install `requirements-mlx.txt` in the Python environment. Model weights are separate downloads governed by their own licenses. `model-manifest.json` pins the included models and notices; `python scripts/provision_models.py --help` lists provisioning options. The Process Models page reports installed and missing components. Additional optional models, including embeddings, must be installed at the paths reported by the catalog. Inference itself does not silently download missing models.

Run `CALL1_BACKEND=mlx python -m call1.launch --demo --handlers real` with the required weights available. The Process demo studio can import its separately attributed 15.5-second AppTek excerpt. Alternatively import recordings you are authorized to process. The historical five macOS-voice recordings are not distributed.

The optional `scripts/generate_test_audio.py` creates non-speech test tones for fake-handler tests. These are not recordings for real-model evaluation. Remove those generated WAVs before a real-model evaluation; their marker prevents the demo launcher from automatically importing them into a real run.

## Regular local installation

`python -m call1.launch` starts Store and Process, runs migrations, and issues a scoped service key when required. Run `python -m call1.store setup-code --email <your-email> --display-name <your-name>` to enroll the initial administrator, using the one-time code and passkey flow. Normal reviewer authentication uses passkeys. The demo's persona sign-in is separate.

Store serves Evaluate at `/` and its operations console at `/console/`. Process's console is for trusted local operators. Service keys and console tokens are credentials, not shareable demo links.

## Data and configuration

Keep `data/`, model weights, logs, local configuration, uploaded recordings and credentials out of Git. Store is the owner of persistent data; Process keeps its own configuration and temporary processing state. The Store and Process module READMEs describe individual configuration variables and host commands.

The checked-in setup is a local source application. No macOS app bundle, DMG, bundled Python interpreter, FFmpeg binary, SeaweedFS binary or hosted service is included. Install runtime dependencies separately. The previous native application's installation and backup commands do not apply to this distribution.
