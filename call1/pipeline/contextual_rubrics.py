"""Versioned class scorecards: contextual claims require contextual evidence."""
from call1.models.schemas import CheckType, RubricCheck


STANDARD_GUIDANCE = {
    'REG-01': (
        'The agent explicitly tells the caller during the opening that this call is or may be recorded. A negated statement is not a disclosure.',
        'The agent explicitly contradicts a supplied recording policy. Missing or ambiguous wording alone requires review.',
        'The agent explicitly establishes that recording does not occur on this call.',
    ),
    'SEC-01': (
        'The required verification policy is supplied and the conversation demonstrates its completion before protected account information is discussed. Asking to verify or mentioning an identifier is insufficient. Without that policy choose needs_review.',
        'A supplied verification policy clearly applies and protected account information is explicitly shared before its required steps. Without policy or a clear sequence choose needs_review.',
        'The conversation explicitly concerns general public information only, with no protected account access or transaction.',
    ),
    'COMP-01': (
        'The applicable transaction and required disclosures are supplied, and the agent communicates each required item. Mentioning fees, policy, or cancellation alone is insufficient.',
        'An explicitly supplied required term contradicts what the agent states for this transaction. Missing policy or applicability requires review.',
        'The conversation explicitly establishes that no transaction or applicable terms are involved.',
    ),
    'ETIQ-01': (
        'At the end of the conversation the agent both offers further assistance and closes professionally. An opening greeting or mid-call offer is insufficient.',
        'The agent explicitly refuses further assistance or uses a clearly unprofessional closing at the end. An incomplete ending or uncertain call boundary requires review.',
        'The conversation explicitly ends through an agreed transfer to another agent before a closing is appropriate.',
    ),
}


def contextual_presets(legacy):
    result = []
    for original in legacy:
        rubric = original.model_copy(deep=True)
        rubric.rubric_id = original.rubric_id.removesuffix('_v1') + '_v2'
        rubric.name = original.name + ' — Contextual v2'
        rubric.description = 'Class QA template requiring contextual evidence and human review when policy or applicability is unknown. Not a compliance certification.'
        for criterion in rubric.criteria:
            guidance = STANDARD_GUIDANCE.get(criterion.criterion_id, (
                f'The conversation clearly establishes this complete behavior: {criterion.description} Do not infer it from isolated keywords. Required business or regulatory policy must be supplied; otherwise choose needs_review.',
                'Explicit conversation evidence contradicts the required behavior under a supplied applicable policy. Missing policy, uncertain applicability, or incomplete evidence requires review.',
                'Explicit conversation evidence establishes that this criterion does not apply. Silence alone is insufficient.',
            ))
            criterion.check = RubricCheck(
                check_type=CheckType.SEMANTIC_JUDGEMENT, speaker=criterion.check.speaker,
                requires_policy=criterion.category in ('COMPLIANCE', 'SECURITY') and criterion.criterion_id != 'REG-01',
                pass_when=guidance[0], fail_when=guidance[1], not_applicable_when=guidance[2],
            )
        result.append(rubric)
    return result


# --- Retail demo policy --------------------------------------------------------------------------
# SEC-01 and COMP-01 set ``requires_policy``: without a configured business policy they are FLAGGED
# ("requires a configured business policy") and no call can pass. The shipped default keeps them
# empty on purpose, because an install must state its own policy. ``python -m call1.launch --demo``
# (and ``scripts/apply_demo_policy.py`` for a running demo) publishes the next version of
# ``call1_standard_v2`` with this retail contact-centre policy text (``call1.demo_setup``).
# Definition text: no numbers or caller details (Store's text detectors refuse those).

RETAIL_DEMO_POLICY = {
    'SEC-01': (
        'Identity verification policy (retail contact centre). Before the agent discusses, changes or refunds a '
        'specific order, account, payment or loyalty balance, the caller is verified with two identifiers from this '
        'list: full name, order number, the email address or phone number on the account, account number, or date '
        'of birth. The agent asks for the identifiers and acknowledges them (for example "thank you for verifying") '
        'before sharing any order or account detail. General questions about products, stock, prices, store hours '
        'or store policies need no verification, so for those calls this criterion does not apply.'
    ),
    'COMP-01': (
        'Required disclosures policy (retail contact centre). Recording: before any order or account detail is '
        'discussed, the agent states that the call is or may be recorded. Transactions: when the agent places an '
        'order, takes a payment, or starts a return, refund, exchange or cancellation on the call, the agent also '
        'states the key terms before the caller agrees: the amount (or refund amount) and how it is paid or paid '
        'back, and any return window, restocking fee or delivery time that applies. A call with no order, payment, '
        'return, refund, exchange or cancellation needs only the recording statement; when that statement was made, '
        'this criterion passes.'
    ),
}


def with_policy(definition: dict, policy: dict = RETAIL_DEMO_POLICY) -> tuple:
    """(definition, changed): a copy of a RubricDefinition JSON object whose criteria named in
    ``policy`` carry that ``check.policy_context``. Criteria that already carry the same text, and
    criteria not in ``policy``, are unchanged. Plain JSON in and out, so HTTP callers need no models."""
    import copy

    result = copy.deepcopy(definition)
    changed = False
    for criterion in result.get('criteria') or []:
        text = policy.get(criterion.get('criterion_id'))
        check = criterion.get('check')
        if text is None or not isinstance(check, dict):
            continue
        if (check.get('policy_context') or '').strip() != text:
            check['policy_context'] = text
            changed = True
    return result, changed


def with_demo_recording_weight(definition: dict) -> tuple:
    """Keep disclosure visible without making it an automatic demo-call failure.

    Five points also avoid the former score-only failure: with a threshold of
    eighty, losing twenty-five points failed a call even with critical=False.
    Other criteria and the actual evidence assessment remain unchanged.
    """
    import copy

    result = copy.deepcopy(definition)
    changed = False
    for criterion in result.get('criteria') or []:
        if criterion.get('criterion_id') != 'REG-01':
            continue
        if criterion.get('critical') is not False or criterion.get('weight') != 5.0:
            criterion.update(critical=False, weight=5.0)
            changed = True
    return result, changed
