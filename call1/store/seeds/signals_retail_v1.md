# Retail seed v1 — Contact Signals v2

`signals_retail_v1.json` is a `SignalTaxonomySave` payload (§7.2: `taxonomy`, `expected_record_version`,
`notes`) — the shape the doc defines for saving/publishing a taxonomy draft. `expected_record_version`
is set to `1`, assuming this is applied as the first admin save after Store's install-time seeding of
version 1 (built-ins only, §7.2 "Install-time version 1", §9.1).

Built from `retail_taxonomy_draft.json`/`.md` (30 AppTek retail transcripts, text only). The draft
was a working file and is not in the repo.

**Applying it.** `python -m call1.store apply-signals-seed call1/store/seeds/signals_retail_v1.json`
(`--pipeline v2` also sets the pipeline) publishes it as an audited save; `--demo` uses it. Installs
start with the built-ins only (decision 22). A seed that is already current is a no-op.

**Contract 1.3.0 fixes (F2).** Speakers are the `SpeakerRole` values (`CALLER`, `AGENT`) and every
example is a plain string, as the final contract requires; `tests/store/test_signals_taxonomy.py`
and `tests/test_contracts.py` check that the file passes the save validator (caps and the
definition-text detectors).

## 1. Seed contents

| Category | Type | Speaker | Subcats | Fields | Calls (draft) |
|---|---|---|---:|---:|---:|
| intent | built-in | caller | 9 | 15 | 30 (100%) |
| issue | built-in | caller | 6 | 3 | 7 (23%) |
| friction | built-in | caller | 6 | 0 | 7 (23%) |
| fix_proposed | built-in | agent | 6 | 0 | 20 (67%) |
| agent_reports_completed | built-in | agent | 4 | 0 | 20 (67%) |
| caller_confirms_resolved | built-in | caller | 2 | 0 | 28 (93%) |
| caller_reports_unresolved | built-in | caller | 1 | 0 | 1 (3%) |
| deferred | built-in | agent | 3 | 0 | 15 (50%) |
| **upsell_attempt** | custom | agent | 4 | 4 | 10 (33%) |
| **agent_conduct_concern** | custom | agent | 3 | 0 | 3 (10%) |

Totals: 8 built-in + 2 custom = 10 categories (cap: ≤16 custom, §9.6 — well inside). 44 subcategories,
29 leaf extraction fields. No category has more than 9 active subcategories (cap: 12, §9.6).

**Excluded:** `competitor_mention` (custom category proposed in the draft) — see §3 below.
**Excluded:** every explicit `other` subcategory the draft proposed — see §2.

## 2. Places I shortened, merged or dropped something, and why

### 2.1 Dropped every explicit "other" subcategory (rule, not judgment)

The draft gave each category an explicit `other` subcategory (usually with 0 support). §7.2's
`RESERVED_SUBCATEGORY_IDS = frozenset({"other", "not"})` reserves those IDs — an admin cannot create a
subcategory literally called `other`, because stage 2 already synthesizes "Other <category>" and
"Not <category>" options automatically (§4.1). So I removed all of them; nothing is lost, since an
unmatched span already falls into the system's own Other bucket. This also satisfies the task's
instruction to drop the courier/newspaper-subscription intents into Other rather than giving them a
subcategory — they now have no subcategory at all in `intent`, which *is* dropping them into Other.

### 2.2 Dropped every field whose PII class the contract forbids (rule, not judgment)

§5.2 lists the allowed field `pii_class` values (`none`, `organization`, `product`, `amount`, `date`,
`agent_name`) and the forbidden ones, refused at save with `validation_failed`
(`caller_name`, `account_number`, `card_number`, `phone`, `email`, `address`, `url`, `secret`,
`government_id`) — "The engine never sees them, and they must never be stored as values." That is a
harder rule than "mark it withheld": these fields cannot exist in the taxonomy at all. Dropped:

| Field (draft) | Subcategory | Draft pii_class | Why dropped |
|---|---|---|---|
| `card_number` | order_placement_over_phone | masked_pii | maps to forbidden `card_number` |
| `card_expiry_and_cvv` | order_placement_over_phone | masked_pii | maps to forbidden `secret`/`card_number` |
| `customer_name` | order_placement_over_phone | masked_pii | forbidden `caller_name` |
| `email_address` | order_placement_over_phone | masked_pii | forbidden `email` |
| `phone_number` | order_placement_over_phone | masked_pii | forbidden `phone` |
| `mailing_address` | order_placement_over_phone | masked_pii | forbidden `address` |
| `loyalty_number` | loyalty_points_question | masked_pii | forbidden `account_number` |

`order_placement_over_phone` goes from 8 draft data points to 2 (`payment_method`, `order_total`).
This is the single biggest cut in the seed, and it is also the draft's own flagged item: 5+ of the 30
calls have a full card number read aloud, which is exactly why that field can never be an extraction
target — Call1 must never store it, only mask it (invariant in `call1/CLAUDE.md`: "Call1 never sees
customer data").

### 2.3 Mapped the draft's ad hoc `pii_class` values onto the contract's enum (judgment)

The draft used three informal classes (`none`, `business`, `masked_pii`) that don't exist in §5.2's
`FieldPiiClass`. I mapped case by case:

- `order_or_receipt_number` (draft: `business`) → `none`. The draft's own note argues it's an opaque
  lookup key that doesn't identify the customer alone — I agree, and `none` is the closest fit since
  there's no generic "business identifier" class in the contract. Flagged as ambiguity 3 below.
- `store_location` (draft: `business`) → `organization` (a place/branch name, not a customer detail).
- `program_name`, `competitor_name` (draft: `business`/`none`) → `organization`.
- `addon_item`, `item_description` (draft: `none`) → `product` (the doc's class exists for exactly this).
- All `money`-typed fields (draft type `money`) → contract type `amount`, `pii_class: amount`.
- `restock_eta` (draft type `date`) → `pii_class: date`.
- Everything else with no real sensitivity → `none`.

### 2.4 Renamed three IDs to fit the 40-character node-ID cap (mechanical)

`SIGNAL_NODE_ID_PATTERN` allows at most 40 characters total (§7.2). Three draft IDs measured over that:

| Draft id | Len | Renamed to | Len |
|---|---:|---|---:|
| `offer_alternate_store_ship_or_special_order` | 43 | `offer_alternate_store_or_special_order` | 38 |
| `pending_supplier_restock_or_special_order` | 41 | `pending_supplier_restock_or_order` | 33 |
| (name, not id) `Dismissive or skeptical tone toward caller` | 42 chars (name cap 40) | `Dismissive or skeptical tone` | 29 |

### 2.5 Every gloss rewritten to ≤ 40 characters and generic (rule)

§9.6 caps custom/admin-authored option glosses at 40 characters (`max_option_gloss_chars`), and per
§7.2 that applies to every subcategory gloss (subcategories are always admin-created — the built-ins
constant has no subcategories of its own). All 44 subcategory glosses in this seed were rewritten to
generic, non-quote phrasing at or under 40 characters. Longest: `joke_about_sensitive_data_collection`
at 37. Full count script output: **max subcategory gloss length in the seed = 37 chars; max custom
category gloss = 31 chars** (`upsell_attempt`, cap 40) — both under cap. Every gloss was counted
programmatically, not eyeballed (see §4 below).

### 2.6 Examples rewritten as generic phrases, not call quotes (rule + judgment)

§9.4 requires taxonomy text — glosses, descriptions, **examples**, field names/descriptions — to carry
no caller details, and only "describe the behavior" ("Names are the gap the rules can't close" — the
automatic detector only catches structured PII like SSNs/cards, not names or other identifying content
in free text). The draft's `examples` were verbatim call quotes (some with names, e.g. "Jeffrey
Walkheart", or exact phrasing tied to one call). I replaced every example with a short, generic,
paraphrased phrase in the style of the doc's own demo example ("cancel", §15) — none are copied from
any transcript.

### 2.7 Custom category descriptions trimmed to fit the 240-character cap (mechanical)

`agent_conduct_concern`'s description needed two trims to land at 228 characters (cap 240, §7.2)
after removing operational asides that pushed it over.

## 3. Judgment call: `competitor_mention` excluded

The draft proposed `competitor_mention` at 2/30 calls (7%), explicitly "thin — include with caution."
I chose to exclude it from this seed rather than include it as a third custom category:

- One of its two examples ("vague elsewhere is cheaper") carries no concrete extraction value and is
  really a weaker restatement of `price_or_promo_question > price_match_requested`, which is already
  built-in-adjacent and better supported (11/30 calls).
- Two calls is below a reasonable bar for a customer-visible category name that other reviewers will
  see on every call — precision at that support level can't be estimated (§9.5: precision is shown
  "only once at least 5 hits have been judged").
- It is analytics/pricing-nuance, not a QA/compliance signal, so the cost of waiting for more data is
  low.

By contrast I kept `agent_conduct_concern` at similarly thin support (3/30, 10%) because it is a
compliance/QA-relevant behavior category (rudeness, joking about sensitive data collection) where the
cost of *not* flagging a rare occurrence is asymmetrically high for a call-QA product, and I marked its
category description "Experimental" and named the sample size directly in the seeded text so admins see
the caveat in the editor. This is a judgment call, not a rule from the doc — flagging it for John/F1 as
ambiguity 4 below.

## 4. Cap and rule verification

Ran programmatically (`build_seed.py` in this folder), not eyeballed:

| Cap / rule | Doc ref | Result |
|---|---|---|
| Node ID pattern `^[a-z0-9][a-z0-9_-]{0,39}$` | §7.2 | all category/subcategory/field IDs pass |
| `other`/`not` reserved subcategory IDs | §7.2 | none used |
| Category name/gloss ≤ 40 chars | §7.2 | all pass; max custom gloss 31 chars |
| Subcategory name ≤ 40, gloss ≤ 40, description ≤ 240 | §7.2, §9.6 | all pass; max gloss 37 chars |
| Field name ≤ 40, description ≤ 200 | §7.2 | all pass |
| Enum values 1–40 chars, ≤ 12 per field | §7.2 | all pass (max 7 values, `payment_method`) |
| Fields per category+subcategory path ≤ 12 | §9.6 | all pass; max 4 (`check_stock_availability`) |
| Active subcategories per category ≤ 12 | §9.6 | all pass; max 9 (`intent`) |
| Active custom top-level categories ≤ 8 (≤16 total) | §9.6 | 2 custom, well under |
| Field `pii_class` never in `FORBIDDEN_FIELD_PII_CLASSES` | §5.2 | verified — 7 fields dropped instead |
| No PII in definition text (names/numbers/emails/quotes) | §9.4 | all examples/descriptions rewritten generic; no call quotes remain |

## 5. Doc sections used for each shape

| Element | Defined in |
|---|---|
| Overall save payload shape (`taxonomy`, `expected_record_version`, `notes`) | §7.2 `SignalTaxonomySave` |
| `SignalCategory` / `SignalSubcategory` / `SignalField` shapes and field names | §7.2 |
| Built-in category fixed name/gloss/speaker text | §7.2 `BUILTIN_SIGNAL_CATEGORIES` |
| Which category fields an admin may edit on a built-in | §7.2 "Built-ins are fixed" |
| Stage-2 options (subcategory gloss = the option text, "Other"/"Not" synthesized) | §4.1, §4.2 |
| Reserved subcategory IDs | §7.2 |
| Field types and PII classes, forbidden list | §5.2 |
| Caps (custom categories, subcategories, fields, gloss length) | §9.6 |
| Definition-text PII rule | §9.4 |
| Install-time version 1 / no-drafts / one-document-versioned-whole rules | §7.2, §9.1 |
| Demo-style generic example convention | §15 |

## 6. Ambiguities for the F1 contract author to settle

1. **Built-in category `description` field is unaddressed.** §7.2's "Built-ins are fixed" list of
   editable fields (`threshold`, `subcategory_threshold`, `subcategories`, `fields`, `narrow_quote`,
   `examples`) omits `description`, and `BuiltinCategory` (the fixed-text constant) has no
   `description` slot either — only `name`, `gloss`, `speaker`. So it's unclear whether an admin/seed
   may set a built-in category's `description` at all, whether it defaults to `null` forever, or
   whether Store fills it from somewhere else. I left it `null` on every built-in in this seed. F1
   should confirm whether that's correct or whether `description` should join the editable set.
2. **`ExampleText`'s own length/format constraints are undefined in the doc.** §7.2 gives
   `examples: List[ExampleText]` a list-length cap (5) but the doc never shows `ExampleText`'s own
   field definition (unlike `EnumLabel`, which is called out as "1–40 chars" inline). I treated
   examples as short free-text phrases (all well under 60 characters here) but F1 should pin an actual
   min/max length and confirm examples go through the same §9.4 PII-detector pass as glosses/descriptions
   (the doc lists "examples" among the PII-scanned text paths, so presumably yes — but the type itself
   isn't specified).
3. **No PII class exists for "business-internal but non-identifying" fields** like an order/receipt
   number. §5.2's allowed set (`none`, `organization`, `product`, `amount`, `date`, `agent_name`) has no
   analog to the draft's informal `business` class. I used `none` for `order_or_receipt_number`, matching
   the draft's own reasoning that it doesn't identify the customer alone — but F1 should confirm `none`
   is the intended bucket for this pattern generally (loyalty/order/ticket numbers that are opaque
   lookup keys), versus adding a dedicated PII class in a future contract revision.
4. **Thin-support custom categories: no numeric bar in the doc.** §9.6 caps *how many* custom
   categories an install may have, but nothing in §7–§9 sets a minimum support threshold for *shipping*
   one in a seed (only §9.5's "precision shown at ≥5 judged hits", which is a display rule, not a save
   rule). I excluded `competitor_mention` (2/30) and kept `agent_conduct_concern` (3/30) on a judgment
   call about compliance value, documented in §3 above — a bar as low as "any call-supported example"
   is enough to pass validation. F1/John may want an explicit minimum-support convention for future
   customer-submitted taxonomies, since Store's validator has no way to enforce one today.
5. **Whether `agent_conduct_concern` belongs as a shipped custom category vs. a documented exclusion**
   is itself unresolved — I made the call in §3, but it is genuinely close, and reasonable people could
   cut it like `competitor_mention`. Flagging for John's sign-off, per the repo's `CLAUDE.md`: "Stop and ask
   John before changing a decision recorded in StrategyAlignment" — this isn't a StrategyAlignment
   decision, but it is exactly the kind of taxonomy judgment call that doc expects S0/admins to make
   with a preview in hand, not a one-shot seed file.

## 7. Not carried over from the draft (out of scope for a taxonomy save)

The draft's `excluded_zero_support_candidates`, `coverage`, `data_quality_caveats`, and the two
non-retail-call notes are draft/analysis artifacts, not taxonomy content — they don't have a home in
`SignalTaxonomySave` and are intentionally left out of `signals_retail_v1.json`. The two non-retail calls
(courier, newspaper subscription) are reflected only as an absence: no subcategory was created for them,
and their content falls to `intent`'s automatic "Other" (§2.1 above).
