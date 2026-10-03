"""Real stage handlers: the pre-split analysis pipeline behind the handler interface.

``build_registry("real")`` imports this package and calls ``register``. Every model-backed stage
wraps the existing modules (``call1.pipeline.*``, ``call1.adapters.*``, ``call1.summarizer``,
``call1.question_models``, ``call1.qa_output``, ``call1.redaction``) so the split produces what the
legacy app produced, as contract artifact content:

==============================  ===========================================================  ============
job type                        wraps                                                        module
==============================  ===========================================================  ============
``validation_vad``              ``AudioValidator`` + ``VoiceActivityDetector``                ``media``
``asr``                         Parakeet TDT v3 (``CALL1_BACKEND=mlx``) or faster-whisper     ``media``
                                + the vocabulary pass (Whisper Small) and rule merge          ``vocabulary``
``speaker_attribution``         Nemotron-3-Diarization + cluster rule (mono); corrections      ``media``
``acoustic_tone``               MERaLiON SER seven-second speaker blocks                     ``analysis``
``text_sentiment``              Cardiff RoBERTa per turn                                     ``analysis``
``enrichment``                  ``extract_numeric_references``                               ``analysis``
``qa_deterministic``            ``RubricEvaluator`` checks + grounding guardrail             ``qa``
``qa_criterion``/``_escalation``  the ``QuestionRouter`` path for one model, quote-verified  ``qa``
``summary_segment``/``_synthesis``  ``call1.summarizer`` chunk and synthesis prompts         ``summary``
``summary_assembly``            the whole-call citation check (replaces the code stage)      ``summary``
``contact_signals_*`` passes    one ``extract_contact_signals`` pass each                    ``signals``
``contact_signals_categorize``  Contact Signals v2 stage 1 on Gemma (decision 24)            ``signals_v2``
``..._subcategorize``           stage 2 on Gemma (decision 24)                               ``signals_v2``
``contact_signals_extract``     stage 3 on Gemma, token-budgeted batches, in-job fallback    ``signals_v2``
==============================  ===========================================================  ============

``embeddings`` is Nemotron-3-Embed-1B (``handlers/embeddings.py``, the ``call1.embedding`` module
Store embeds queries with; the fake embedder when ``CALL1_EMBEDDING_BACKEND=fake``). ``qa_scorecard``
and ``contact_signals_merge`` stay the code stages ``handlers/code.py`` registers in every mode.

Model runtimes (MLX, torch, faster-whisper) are imported only inside ``run``. Every MLX job shares
the worker's single ``mlx`` slot, and the legacy ``inference_lock`` still serializes local
inference inside the process. ``registry.entry_status`` reports what this host can serve
(``paths.entry_status``); the worker releases a claim for an entry that is not available here
before any inference starts.
"""

from __future__ import annotations

import logging
from typing import Any, List

from call1.process.handlers.base import Handler

from .paths import backend_label, entry_detail, entry_status

log = logging.getLogger("call1.process.handlers.real")


def real_handlers(config: Any = None, catalog: Any = None) -> List[Handler]:
    from .analysis import RealAcousticTone, RealEnrichment, RealTextSentiment
    from .media import RealAsr, RealSpeakerAttribution, RealValidationVad
    from .qa import RealQaCriterion, RealQaDeterministic, RealQaEscalation
    from .signals import RealLifecyclePass, RealResolutionPass
    from .signals_v2 import RealSignalsCategorize, RealSignalsExtract, RealSignalsSubcategorize
    from .summary import RealSummaryAssembly, RealSummarySegment, RealSummarySynthesis

    batch = int(getattr(config, "summary_batch_turns", 60) or 60)
    return [
        RealValidationVad(), RealAsr(), RealSpeakerAttribution(), RealAcousticTone(), RealTextSentiment(), RealEnrichment(),
        RealQaDeterministic(), RealQaCriterion(), RealQaEscalation(), RealSummarySegment(batch), RealSummarySynthesis(),
        RealSummaryAssembly(), RealLifecyclePass(), RealResolutionPass(),
        RealSignalsCategorize(), RealSignalsSubcategorize(), RealSignalsExtract(catalog),
    ]


def register(registry: Any, context: Any) -> None:
    """The ``build_registry`` hook: one handler per job type, and this host's entry status. A
    handler module that fails to import is reported in the console's notes and registers nothing,
    so its job types wait for a Process that has them; it never stops Process."""
    try:
        handlers = real_handlers(getattr(context, "config", None))
    except Exception as exc:
        log.exception("real handlers failed to load")
        registry.notes.append(f"Real handlers failed to load ({type(exc).__name__}); only the code stages run here.")
        return
    for handler in handlers:
        if hasattr(handler, "catalog") and getattr(handler, "catalog", None) is None:
            handler.catalog = getattr(context, "catalog", None)  # the stage-3 fallback entry lookup
        registry.register(handler)
    from call1.process.handlers.embeddings import embeddings_handler

    registry.register(embeddings_handler("real"))
    registry.entry_status = entry_status
    registry.entry_detail = entry_detail
    registry.notes.append(f"Real handlers ({backend_label()}): the pre-split pipeline modules; weights under CALL1_MODELS_DIR.")


__all__ = ["real_handlers", "register"]
