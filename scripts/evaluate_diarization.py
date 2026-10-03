"""Exact interval DER with optimal global speaker mapping, overlap included, no collar.

This small local metric is not the upstream benchmark protocol. Reference labels
must be independently suitable; the metric cannot validate their accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path


def diarization_errors(reference, hypothesis, duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be positive and finite.")
    events = defaultdict(list)
    events[0.0]
    events[duration]
    speakers = [set(), set()]
    for side, segments in enumerate((reference, hypothesis)):
        for segment in segments:
            start, end = float(segment['start']), float(segment['end'])
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError("Segments require finite, ordered timestamps.")
            start, end = max(0.0, start), min(duration, end)
            if end <= start:
                continue
            speaker = str(segment['speaker'])
            speakers[side].add(speaker)
            events[start].append((side, speaker, 1))
            events[end].append((side, speaker, -1))
    if max(map(len, speakers)) > 8:
        raise ValueError("This exhaustive assignment metric supports at most eight speakers.")
    active = [Counter(), Counter()]
    intervals = []
    previous = 0.0
    for boundary in sorted(events):
        if boundary > previous:
            intervals.append((boundary - previous, set(active[0]), set(active[1])))
        for side, speaker, delta in events[boundary]:
            active[side][speaker] += delta
            if not active[side][speaker]:
                del active[side][speaker]
        previous = boundary

    refs, hyps = map(sorted, speakers)
    width = max(len(refs), len(hyps))
    overlap = defaultdict(float)
    reference_time = missed = false_alarm = comparable = 0.0
    for seconds, ref, hyp in intervals:
        reference_time += seconds * len(ref)
        missed += seconds * max(0, len(ref) - len(hyp))
        false_alarm += seconds * max(0, len(hyp) - len(ref))
        comparable += seconds * min(len(ref), len(hyp))
        for r in ref:
            for h in hyp:
                overlap[r, h] += seconds
    best_correct = -1.0
    best_mapping = {}
    for assignment in itertools.permutations(range(width), len(hyps)):
        mapping = {h: refs[i] if i < len(refs) else None for h, i in zip(hyps, assignment)}
        correct = sum(overlap[r, h] for h, r in mapping.items() if r is not None)
        if correct > best_correct:
            best_correct, best_mapping = correct, mapping
    confusion = max(0.0, comparable - max(0.0, best_correct))
    return {
        'protocol': 'Exact interval DER, optimal global one-to-one mapping, overlap included, zero collar; clipped to recording duration.',
        'reference_speaker_seconds': reference_time,
        'missed_speaker_seconds': missed, 'false_alarm_speaker_seconds': false_alarm,
        'confused_speaker_seconds': confusion,
        'der': (missed + false_alarm + confusion) / reference_time if reference_time else None,
        'hypothesis_to_reference_mapping': best_mapping,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('sample_audio/telephony_training_set/manifest.json'))
    parser.add_argument('--call-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing evaluation.')
    benchmark = json.loads(args.benchmark.read_text())
    call = json.loads(args.manifest.read_text())[args.call_id]
    if benchmark['audio'] != call['file_name']:
        parser.error('Benchmark audio does not match the selected reference call.')
    reference = [{'start': t['start_time'], 'end': t['end_time'], 'speaker': t['speaker_id']} for t in call['turns']]
    result = diarization_errors(reference, benchmark['segments'], benchmark['duration_seconds'])
    result.update({'call_id': args.call_id, 'benchmark': str(args.benchmark),
                   'scope': 'Known public role-played development recording; reference annotations not independently verified.'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as target:
        json.dump(result, target, indent=2)
        target.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
