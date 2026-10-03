"""Anonymous speaker clustering helpers; acoustic clusters never imply roles.

Segments come from the diarization model on the source clock (``speaker`` is a 0-based cluster
index, up to eight with Nemotron-3-Diarization) and become ``speaker_1`` ... ``speaker_8``."""
from bisect import bisect_left
from collections import defaultdict

from call1.models.schemas import SpeakerRole


def cluster_lookup(segments):
    """``cluster(start, end)``: the anonymous speaker cluster covering a time span, or None when
    coverage is weak or a second speaker overlaps materially (the abstention rule)."""
    spans = sorted(segments, key=lambda s: s['start'])
    starts = [s['start'] for s in spans]
    maximum_ends = []
    for span in spans:
        maximum_ends.append(max(span['end'], maximum_ends[-1] if maximum_ends else 0))

    def cluster(start, end):
        if end <= start:
            return None
        seconds = defaultdict(float)
        i = bisect_left(starts, end) - 1
        while i >= 0 and maximum_ends[i] > start:
            span = spans[i]
            overlap = max(0, min(end, span['end']) - max(start, span['start']))
            seconds[span['speaker']] += overlap
            i -= 1
        ranked = sorted(seconds.items(), key=lambda item: item[1], reverse=True)
        # Abstain on weak coverage or material overlapping/ambiguous speakers.
        if not ranked or ranked[0][1] < .6 * (end - start):
            return None
        if len(ranked) > 1 and ranked[1][1] > .2 * (end - start):
            return None
        return f'speaker_{int(ranked[0][0]) + 1}'

    return cluster


def turn_clusters(turns, segments):
    """One cluster (or None) per whole turn, keyed by turn_id, with the same abstention rule.

    The split Process app fixes turn IDs at ASR, so its speaker attribution labels whole turns and
    never splits one; ``attach_clusters`` below splits turns at word-level cluster changes."""
    cluster = cluster_lookup(segments)
    return {turn.turn_id: cluster(turn.start_time, turn.end_time) for turn in turns}


def attach_clusters(turns, segments):
    cluster = cluster_lookup(segments)

    output = []
    for turn in turns:
        words = turn.word_timestamps or []
        joined = ''.join(w.word for w in words)
        # Incomplete alignment must not delete spoken text during splitting.
        if not words or ''.join(joined.split()) != ''.join(turn.text.split()):
            output.append(turn.model_copy(update={
                'speaker': SpeakerRole.UNKNOWN,
                'speaker_cluster': cluster(turn.start_time, turn.end_time),
            }))
            continue
        groups = []
        for word in words:
            label = cluster(word.start_time, word.end_time)
            if not groups or groups[-1][0] != label:
                groups.append((label, []))
            groups[-1][1].append(word)
        for label, group in groups:
            text = ''.join(w.word for w in group).strip()
            if not text:
                continue
            output.append(turn.model_copy(update={
                'speaker': SpeakerRole.UNKNOWN, 'speaker_cluster': label,
                'start_time': group[0].start_time, 'end_time': group[-1].end_time,
                'text': text, 'raw_text': text, 'word_timestamps': group,
                # Word probabilities are real only when the ASR scored the turn (Parakeet does not).
                'confidence': (sum(w.probability for w in group) / len(group)) if turn.confidence is not None else None,
            }))
    for index, turn in enumerate(output):
        turn.turn_id = index
    return output
