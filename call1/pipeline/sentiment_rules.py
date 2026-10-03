"""Deterministic comparisons of model estimates; missing evidence requires review."""
import math

from call1.models.schemas import (
    RubricCheck, RubricCriterion, RubricDefinition, RubricVerdict,
    SpeakerRole, VerdictStatus,
)


def evaluate_sentiment(criterion, check, transcript):
    start, end = 0.0, transcript.duration_seconds
    if check.window_seconds:
        if check.window_seconds > 0:
            end = min(end, check.window_seconds)
        else:
            start = max(0.0, end + check.window_seconds)
    text = check.metric == "text_polarity"
    rows = transcript.turns if text else transcript.tone_blocks
    eligible, scored = [], []
    for row in rows:
        if check.speaker is not None and row.speaker != check.speaker:
            continue
        overlap = min(end, row.end_time) - max(start, row.start_time)
        if overlap <= 0:
            continue
        if text and not row.text.strip():
            continue
        if not text and row.status == "NO_SPEECH":
            continue
        eligible.append(row)
        if row.speaker == SpeakerRole.UNKNOWN:
            continue
        status = (row.text_analysis or {}).get("status") if text else row.status
        value = row.text_sentiment if text else getattr(row, check.metric)
        if status != "SCORED" or value is None or not math.isfinite(value):
            continue
        if not (-1 if text else 0) <= value <= 1:
            continue
        # Text is equal-weighted per turn. Tone is weighted by attributed speech
        # inside the selected window; block estimates themselves are not rerun.
        weight = 1.0 if text else sum(
            max(0.0, min(end, b) - max(start, a)) for a, b in row.speech_intervals
        )
        if weight > 0:
            scored.append((row, value, weight))
    coverage = len(scored) / len(eligible) if eligible else 0.0
    common = dict(criterion_id=criterion.criterion_id, criterion_name=criterion.name,
                  speaker=check.speaker or SpeakerRole.AGENT)
    if len(scored) < check.min_samples or coverage < check.min_coverage:
        return RubricVerdict(**common, status=VerdictStatus.FLAGGED, confidence=0.0,
            reasoning=f"Insufficient {check.metric} evidence: {len(scored)}/{len(eligible)} eligible samples scored "
                      f"({coverage:.0%}); requires {check.min_samples} samples and {check.min_coverage:.0%} coverage. "
                      "Missing, disabled, and failed analysis cannot establish a pass or fail.")
    values = [v for _, v, _ in scored]
    value = (sum(v*w for _, v, w in scored) / sum(w for _, _, w in scored)
             if check.aggregation == "mean" else
             min(values) if check.aggregation == "min" else max(values))
    passed = value >= check.metric_threshold if check.comparison == "gte" else value <= check.metric_threshold
    ids = ", ".join(str(r.turn_id if text else r.block_id) for r, _, _ in scored)
    return RubricVerdict(**common, status=VerdictStatus.PASS if passed else VerdictStatus.FAIL,
        confidence=1.0, timestamp_range=(max(start, min(r.start_time for r, _, _ in scored)),
                                       min(end, max(r.end_time for r, _, _ in scored))),
        reasoning=f"{check.aggregation} {check.metric} = {value:.3f}; required "
                  f"{'>=' if check.comparison == 'gte' else '<='} {check.metric_threshold:.3f}. "
                  f"Scored {len(scored)}/{len(eligible)} eligible {'turns' if text else 'blocks'} "
                  f"({coverage:.0%} coverage); IDs: {ids}. "
                  f"{'Equal weight per turn' if text else 'Mean weighted by attributed speech seconds'}. "
                  "Window includes overlapping samples using their full model estimate. "
                  "This is a comparison of model estimates, not verified emotion; confidence describes the comparison.")


def sentiment_presets():
    def rule(id, name, metric, threshold, speaker=SpeakerRole.AGENT, comparison="gte", window=None):
        return RubricCriterion(criterion_id=id, name=name, category="QUALITY", weight=50,
            description="Editable coaching threshold; calibrate against reviewed calls before operational use.",
            check=RubricCheck(check_type="sentiment_metric", metric=metric,
                metric_threshold=threshold, comparison=comparison, speaker=speaker,
                window_seconds=window, min_samples=2, min_coverage=.8))
    return [
        RubricDefinition(rubric_id="call1_agent_sentiment_coaching_v1", name="Agent sentiment coaching",
            category="CUSTOMER_CARE", description="Model-based coaching: agent text polarity and acoustic valence. Starter thresholds, not compliance standards.",
            criteria=[rule("agent_text", "Agent language is neutral or positive on average", "text_polarity", 0),
                      rule("agent_valence", "Agent tone meets the valence target", "valence", .45)]),
        RubricDefinition(rubric_id="call1_caller_closing_sentiment_v1", name="Caller closing sentiment",
            category="CUSTOMER_CARE", description="Caller experience signals in the final 60 seconds. Does not attribute caller emotion to agent performance.",
            criteria=[rule("caller_text", "Closing caller text is neutral or positive", "text_polarity", 0, SpeakerRole.CALLER, window=-60),
                      rule("caller_valence", "Closing caller valence meets target", "valence", .45, SpeakerRole.CALLER, window=-60)]),
        RubricDefinition(rubric_id="call1_agent_tone_balance_v1", name="Agent tone balance",
            category="CUSTOMER_CARE", description="Acoustic coaching across seven-second blocks. Arousal estimates activation; dominance estimates vocal control, not politeness.",
            criteria=[rule("agent_arousal", "Average activation stays within target", "arousal", .65, comparison="lte"),
                      rule("agent_dominance", "Average vocal control meets target", "dominance", .4)]),
    ]
