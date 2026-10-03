"""The results area: call records and result projections (transcript, QA scorecard, summary,
contact signals), audio playback, semantic search (Nemotron-3-Embed-1B), reviews and their
expected-version writes, the human review queue and its rules (never the processing queue),
reviewer profiles, rubrics (drafts, publish, versions, retire) and metrics.

Owner: the results builder. Files: ``routes.py`` (handlers), ``projections.py`` (the hooks the
queue area calls inside its transactions), ``api.py`` (functions other areas call),
``migrations/040_results.sql`` (and 041-049).
"""
