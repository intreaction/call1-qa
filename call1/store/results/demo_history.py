"""Host-only, deterministic fictional sessions for demo dashboards and call walkthroughs.

No model inference, source recordings or human-review labels are fabricated. These are imported
snapshots with explicit synthetic provenance, not processing jobs. The five audio demos remain
the end-to-end pipeline examples. Store owns all persistence; this never opens the legacy DB.
"""
from __future__ import annotations

from datetime import timedelta, timezone
import random

from call1.contracts.artifacts import ArtifactKind
from call1.contracts.calls import CallMetadata, ConversationRegistration, IngestionKind, SourceKind, SourceReference
from call1.contracts.common import canonical_digest
from call1.contracts.contents import (QaScorecardContent, ScorecardRubricRef, SummaryContent, TranscriptContent,
                                     TranscriptTurnContent, VerdictView)
from call1.contracts.events import ChangeKind
from .. import db, feed
from ..clock import ManualClock
from ..queue import api as queue_api
from . import projections, records, rubric_store

DEFAULT_SESSIONS = 560
AGENTS = ("Avery", "Jordan", "Morgan", "Riley", "Sam", "Taylor", "Casey", "Drew", "Alex", "Jamie", "Quinn", "Skyler")
TOPICS = ("a delayed delivery", "a return", "a damaged item", "stock availability", "a refund", "a billing question", "an exchange", "a missing package")
OPENINGS = {"PASS": "Thanks for calling Demo Retail. This call is recorded for quality assurance.",
            "FAIL": "This call is not recorded, even though our recording policy says it is.",
            "FLAGGED": "Hello, how can I help today?"}
VERIFICATION = {"PASS": "Your two required verification checks are complete. I can now open your account.",
                "FAIL": "I opened your private account details before doing the required verification.",
                "FLAGGED": "Let me check your account."}
POLICY = {"PASS": "Under this demo policy, returns are accepted within thirty days, with no return fee. Refunds take five business days.",
          "FAIL": "I will charge a return fee, although this demo policy requires free returns.",
          "FLAGGED": "The usual policy applies; I have not explained the terms yet."}
CLOSING = {"PASS": "Is there anything else I can help with? Thank you for calling, and have a good day.",
           "FAIL": "I will not help you any further. Goodbye.",
           "FLAGGED": "Let me look into that."}


def seed_demo_history(store, *, count: int = DEFAULT_SESSIONS) -> int:
    if not store.config.demo_mode:
        raise ValueError("Synthetic sessions require CALL1_STORE_DEMO=1; real Store data must not be seeded")
    if not 1 <= count <= 5000:
        raise ValueError("Session count must be between 1 and 5000")
    added = 0
    # Keep timestamps stable when a partial seed is retried or an existing demo is expanded.
    with store.connection() as conn:
        first = conn.execute("SELECT MIN(created_at) AS first FROM results_calls WHERE external_call_ref LIKE 'SYNTHETIC-DEMO-v1-%'").fetchone()["first"]
    anchor = db.parse_ts(first).replace(hour=9, minute=0, second=0, microsecond=0) if first else (
        store.clock.now().astimezone(timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0) - timedelta(days=56))
    with store.connection() as conn:
        rubric = rubric_store.current_version(conn, "call1_standard_v2")
        if rubric is None:
            raise ValueError("Publish call1_standard_v2 before seeding demo history")
        if [c.criterion_id for c in rubric.definition.criteria] != ["REG-01", "SEC-01", "COMP-01", "ETIQ-01"]:
            raise ValueError("Demo history expects the four standard criteria; preserve the customized rubric and use a fresh demo root")
        for i in range(count):
            rng = random.Random(568_000 + i)
            day = i % 56
            when = anchor + timedelta(days=day, minutes=((i // 56) % 14) * 37)
            topic = TOPICS[i % len(TOPICS)]
            # Less frequent misses toward the end give a plausible, explicitly fictional trend.
            miss = .22 - day / 56 * .14
            statuses = ["FAIL" if (r := rng.random()) < miss * .55 else "FLAGGED" if r < miss else "PASS" for _ in rubric.definition.criteria]
            lines = [OPENINGS[statuses[0]], f"I am calling about {topic}.", VERIFICATION[statuses[1]],
                     "Thank you. What are my options?", POLICY[statuses[2]], "I understand the next step.", CLOSING[statuses[3]]]
            # A short scripted session, not an invented multi-minute recording.
            duration = float(sum(len(line.split()) for line in lines) / 2.4 + 12)
            step = duration / len(lines)
            transcript = TranscriptContent(duration_seconds=duration, language="en", is_redacted=True, turns=[
                TranscriptTurnContent(turn_id=j, speaker="CALLER" if j in (1, 3, 5) else "AGENT", start_time=round(j * step, 2),
                                      end_time=round((j + 1) * step, 2), text=line) for j, line in enumerate(lines)])
            critical = any(c.critical and s == "FAIL" for c, s in zip(rubric.definition.criteria, statuses))
            score = round(sum(c.weight for c, s in zip(rubric.definition.criteria, statuses) if s == "PASS") /
                          sum(c.weight for c in rubric.definition.criteria) * 100, 1)
            review = critical or "FLAGGED" in statuses
            scorecard = QaScorecardContent(rubric=ScorecardRubricRef(rubric_id=rubric.ref.rubric_id, rubric_version=rubric.ref.version,
                                                                   digest=rubric.ref.digest),
                overall_score=score, passed=score >= rubric.definition.pass_threshold and not critical and not review,
                critical_failure=critical, requires_human_review=review, evaluated_at=when + timedelta(seconds=duration),
                escalation_reasons=["Synthetic demo scenario needs review"] if review else [],
                verdicts=[VerdictView(criterion_id=c.criterion_id, criterion_name=c.name, status=s, confidence=.55 if s == "FLAGGED" else .94,
                    quoted_evidence=lines[t], quote_turn_id=t, timestamp_range=(round(t * step, 2), round((t + 1) * step, 2)),
                    reasoning=f"Synthetic demo verdict ({s}); scripted for presentation, not produced by a model.")
                    for c, s, t in zip(rubric.definition.criteria, statuses, (0, 2, 4, 6))])
            summary = SummaryContent(narrative=f"Synthetic demo session about {topic}. No source recording or model inference exists for this example.",
                key_points=[f"Scenario: {topic}", "Scripted QA outcomes for dashboard and review demonstrations."],
                route_class="synthetic-demo", catalog_entry_id="synthetic-demo-history-v1", generated_at=when, segments=1)
            registration = ConversationRegistration(ingestion_kind=IngestionKind.CALL_AUDIO,
                source=SourceReference(kind=SourceKind.API_UPLOAD, content_digest=canonical_digest({"demo_history": 1, "index": i}), received_at=when),
                call_metadata=CallMetadata(agent_id=f"demo-agent-{i % 12 + 1:02d}", agent_display_name=f"Demo {AGENTS[i % 12]}",
                    agent_extension=str(201 + i % 12), recorded_at=when, external_call_ref=f"SYNTHETIC-DEMO-v1-{i + 1:04d}",
                    caller_reference="Synthetic caller"))
            original_clock = conn.clock
            conn.clock = ManualClock(when)
            try:
                with db.transaction(conn):
                    conv, artifacts = queue_api.import_demo_snapshot(conn, store, registration, [
                        (ArtifactKind.TRANSCRIPT, transcript), (ArtifactKind.QA_SCORECARD, scorecard), (ArtifactKind.SUMMARY, summary)])
                    if not artifacts:
                        continue
                    for art in artifacts:
                        conn.execute("INSERT INTO results_artifacts (artifact_id, conversation_id, kind, slot, version, checksum, content_type, committed_at) VALUES (?, ?, ?, '', ?, ?, ?, ?)",
                                     (art.id, conv.id, art.kind.value, art.version, art.checksum, art.content_type, db.ts(when)))
                        if art.kind is ArtifactKind.PII_FINDINGS:
                            continue
                        group = {ArtifactKind.TRANSCRIPT: "transcript", ArtifactKind.QA_SCORECARD: "qa", ArtifactKind.SUMMARY: "summary"}[art.kind]
                        conn.execute("INSERT INTO results_group_versions (conversation_id, kind, version, call_id, artifact_id, checksum, state, committed_at) VALUES (?, ?, ?, ?, ?, ?, 'available', ?)",
                                     (conv.id, group, art.version, conv.call_id, art.id, art.checksum, db.ts(when)))
                    projections._update_call(conn, conv.call_id, duration_seconds=duration)
                    qa = next(a for a in artifacts if a.kind is ArtifactKind.QA_SCORECARD)
                    projections._publish_qa(conn, records.call_row(conn, conv.call_id), qa.version, qa.checksum, db.ts(when))
                    feed.append(conn, ChangeKind.CALL, conv.call_id, 1, "synthetic_demo_import", conversation_id=conv.id, call_id=conv.call_id)
                    added += 1
            finally:
                conn.clock = original_clock
    return added
