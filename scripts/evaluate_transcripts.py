"""Score saved ASR checkpoints against supplied public reference transcripts.

This measures normalized word error rate on known examples, not QA accuracy.
No text is sent to a remote service. Reports contain aggregate counts, not text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3


def words(text):
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text.lower().replace('’', "'"), re.UNICODE)


def errors(reference, hypothesis):
    # Each tuple is (distance, substitutions, deletions, insertions).
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for i, expected in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, actual in enumerate(hypothesis, 1):
            if expected == actual:
                current.append(previous[j - 1])
                continue
            d, s, delete, insert = previous[j - 1]
            substitution = (d + 1, s + 1, delete, insert)
            d, s, delete, insert = previous[j]
            deletion = (d + 1, s, delete + 1, insert)
            d, s, delete, insert = current[j - 1]
            insertion = (d + 1, s, delete, insert + 1)
            current.append(min((substitution, deletion, insertion), key=lambda item: item[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {'reference_words': len(reference), 'hypothesis_words': len(hypothesis),
            'substitutions': substitutions, 'deletions': deletions, 'insertions': insertions,
            'word_errors': distance, 'wer': distance / len(reference) if reference else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('sample_audio/telephony_training_set/manifest.json'))
    parser.add_argument('--prefix', default='replay/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    results = []
    with sqlite3.connect(f'file:{args.database.resolve()}?mode=ro', uri=True) as db:
        for source in manifest.values():
            key = args.prefix + source['file_name']
            rows = db.execute('''SELECT a.payload FROM appliance_checkpoints a JOIN ingest_jobs j ON j.id=a.job_id
                WHERE j.s3_key=? AND a.stage='transcription' ORDER BY j.id''', (key,)).fetchall()
            if len(rows) != 1:
                raise ValueError(f'Expected one saved transcription for {key}; found {len(rows)}')
            turns = json.loads(rows[0][0])
            reference = words(' '.join(t['text'] for t in sorted(source['turns'], key=lambda t: t['start_time'])))
            hypothesis = words(' '.join(t.get('raw_text') or t['text'] for t in sorted(turns, key=lambda t: t['start_time'])))
            results.append({'sample_id': source['call_id'], **errors(reference, hypothesis),
                'transcript_turns': len(turns), 'unknown_speaker_turns': sum(t['speaker'] == 'UNKNOWN' for t in turns)})
    totals = {key: sum(row[key] for row in results) for key in ('reference_words', 'hypothesis_words', 'substitutions', 'deletions', 'insertions', 'word_errors')}
    totals['wer'] = totals['word_errors'] / totals['reference_words']
    report = {'samples': len(results), 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'model_manifest': json.loads(Path('model-manifest.json').read_text()), 'totals': totals, 'per_sample': results,
              'method': 'Lowercase Unicode word tokens; punctuation discarded except internal apostrophes; numbers are not expanded. Minimum word-level edit distance. Turns concatenated by start time.',
              'limitations': ['Known public role-played examples; not a held-out QA evaluation.',
                  'Reference transcripts are supplied annotations, not independently corrected audio transcriptions.',
                  'Overlapping speech order, contractions and number formatting can contribute errors.',
                  'Unknown mono speakers do not establish speaker attribution accuracy.',
                  'No team QA labels or summary factuality labels supplied; precision, recall, agreement and factual accuracy are unmeasured.']}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'samples': len(results), 'totals': totals}))


if __name__ == '__main__':
    main()
