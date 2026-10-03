"""On-device training: the customer LoRA (team decision 28; docs/OnDeviceTraining.md).

Reviewers correct the machine in Evaluate (QA verdict overrides, Contact Signals feedback, speaker
corrections) and Store logs each label with IDs, enums and versions only. On the schedule the
customer sets in the Process console (Settings -> On-device training), Process pages those labels
(``listTrainingLabels``), rebuilds each labelled prompt with the engine's own prompt builder and
masking, trains a LoRA over the included model in a subprocess, scores it against the active
adapter on held-out calls, and promotes it only when it is not worse. The labels, the datasets and
the adapter never leave this host; Store sees only the adapter version in attempt provenance.

Open core (Apache 2.0 once the license audit clears): nothing here reads a Pro1 connection or an
entitlement, and nothing is gated.

=====================  =========================================================================
module                 what it holds
=====================  =========================================================================
``settings``           the ``training`` config key: schedule, limits, trainer; validation
``labels``             paging the label log; supersession and withdrawal
``replay``             the source job and artifacts of a label as a ``HandlerJob``
``examples``           the example and held-out item builders, per engine, masked and budgeted
``dataset``            the split by call, dedup, minimums and the JSONL files
``trainer``            ``TrainSpec``, the pluggable trainer (``mlx_lm lora`` or the fake) subprocess
``fake_trainer``       the fake trainer process (no MLX, no torch)
``generate``           the evaluation generation worker process, and ``FakeGenerator``
``evaluate``           scoring answers with the engines' parsers; the promotion rule
``registry``           versioned adapters, the active pointer, rollback, retention, resolution
``runner``             one run's phases, from collecting labels to the decision
``scheduler``          ``TrainingService``: settings, the schedule, the start rule, the claim pause
=====================  =========================================================================
"""
