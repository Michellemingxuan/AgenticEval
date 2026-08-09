"""The three judge prompts. Each states what the LLM decides and what Python decides.

An answer costs three calls, and the split is not arbitrary:

    extract      the question and the answer, nothing else   -> claims
    evidence     claims + the ledger + the run's trace       -> pointers, routes
    eligibility  routes + briefs + earlier turns             -> one verdict

The boundary that matters is between the last two. Everything up to and
including `evidence` is DESCRIPTIVE — what was claimed, what measured it, how
the run got there. Only `eligibility` rules on whether that route answers the
question that was asked. Asked in one breath, a judge that has already decided
a claim is sound describes a route that justifies it, so the description has to
be written before any verdict exists.

The ledger is the expensive payload, ~13k tokens, and it travels exactly once.

Nothing an LLM returns here is trusted on its own. Spans are checked against the
answer, evidence ids and call ids against the run, figures against the payload
they were said to come from, and derivations are recomputed. A pointer that does
not resolve is dropped, not believed.
"""
from __future__ import annotations


EXTRACT_PROMPT = """You split an answer into atomic claims. You do not judge them,
and you are given no evidence — the only question here is what the answer says.

Return JSON with key `claims`. Each claim carries:

    claim_id            c1, c2, … in the order they appear
    answer_span         VERBATIM substring of the answer
    proposition         the claim as one self-contained sentence
    claim_type          quantitative | quantitative_comparison | qualitative |
                        qualitative_comparison | causal | compound |
                        recommendation | uncertainty_or_data_gap | opinion
    is_factual          false ONLY for recommendation / opinion / pure
                        data-gap notes. A statement about METHOD ("commercial
                        cards are identified by Card Portfolio = 'SBS'") IS
                        factual: it is checkable against the call that ran it.
    stance              asserted | attributed_unendorsed | attributed_refuted
    restates_claim_id   id of the earlier claim asserting the same fact, else null
    block_id            the block it came from
    source_locator      for a table cell: {row, column, row_header, column_header}
    numeric_mentions    see below

SPLITTING

Split every independently falsifiable clause, including long opening
sentences. Preserve the subject, period, population, qualifiers and comparison
endpoints in each piece — a fragment that has lost its period is not a claim.
For tables, emit one claim per data cell and copy its row and column into
`source_locator`. Code and headings are not factual.

ANSWER_SPAN

Copy it verbatim; it is checked as a literal substring of the answer. Quote a
table row exactly as it appears, pipes and all. Do not relabel a cell with its
column header and do not reformat "September 2024" as "2024-09". A rebuilt span
is indistinguishable from a claim the answer never made.

NUMERIC MENTIONS

Each is {written, value, unit, comparator, measures, material, quoted}.

`measures` is a short noun phrase naming WHAT the number quantifies, including
the operation and the population: "count of commercial (SBS) cards", "sum of
balance across SBS cards", "unpaid amount in September 2024". This is what the
next pass searches the evidence with. A bare figure cannot be matched to a
field — one tool result routinely states a sum, a row count and a total in the
same sentence.

`material` is TRUE for every number the claim rests on: any measured quantity,
count, amount, rate, share, ratio, denominator or comparison endpoint that
would make the claim false if it changed. A count of ZERO is material. So is a
total that a count is stated "out of". FALSE only for identifiers, ordinals,
list positions, version numbers, and dates or periods used purely as labels —
and for a number that NAMES a metric ("30+ DPD", "90-day past due", "top-5
drivers"), where the measured value is the other number in the sentence. When
in doubt mark it material: an unmarked number is never checked at all, whereas
an over-marked one is merely reported as unverifiable.

Copy hedges verbatim into `written` ("~28+", ">20", "<10") and set `comparator`
to the relation asserted (`>=`, `>`, `<=`, `<`, `==`). Do not round a hedge to
a bare figure — "<10" must arrive as value 10 with comparator "<", so evidence
of 8.2 satisfies it instead of reading as a mismatch.

STANCE

`asserted` when the answer stands behind the claim. `attributed_unendorsed`
when it reports someone else's figure without adopting it. `attributed_refuted`
when it reports a figure AND contradicts it.

An attribution and its refutation are ONE claim. "The report cites 186 returned
payments, but this is not grounded in live specialist output; the payments
table shows none" is a single fact about what the answer concluded. Splitting it
manufactures a claim the answer never made. Mark the disowned figure
`quoted: true` and the verifying figure material.

RESTATEMENTS

An answer often states one fact several ways — in a summary sentence, again as
a bullet, again as "the reports confirm this". Emit each, and set
`restates_claim_id` on the later ones so the fact is counted once.

But a restatement is the SAME assertion worded again, NEVER a narrower one
filed under a broader one. A summary and the values it summarises are separate
facts, because either value can be wrong while the summary still holds: "CDSS
stayed below its trigger all period" is one claim, and "CDSS was 8 in
September" and "CDSS was 9 in May" are two more, not restatements of it. Same
for "the bureau profile stayed clean" over the FICO scores and delinquency
counts underneath it. Per-period, per-entity and per-metric values each stand
on their own.

Two claims are the same fact only when a different thing could NOT make each
false: "357 attempts" and "all 357 succeeded" are distinct; "no returns", "0 of
357 were returned" and "the reports document zero returns" are one fact.

When unsure, leave `restates_claim_id` unset. A folded claim is never verified
at all, so over-folding hides facts AND flatters every rate computed over the
ones that remain.

BEFORE YOU RETURN

Re-read the answer against your own list. Add any independently falsifiable
clause or material number you missed — check every table data cell and every
clause of every long sentence. Correct any `material` flag that is plainly
wrong. Every claim must be a proposition the ANSWER asserts about the domain;
"the value '30+ DPD' is a label" is a statement about these instructions and
must never appear as a claim. Return JSON only."""


EVIDENCE_PROMPT = """You locate the evidence behind claims that have already been
extracted. You do not judge whether the route was the right way to answer the
question — a later pass does that, and guessing at it here biases the evidence
you cite toward the claim you have already decided is fine.

You are given the question, the answer, the claims, the evidence ledger, and
the run's own trace: the team the orchestrator assembled, the brief it gave
each specialist, the operations each ran with their arguments, and the findings
each returned.

Return ONE JSON object with three keys: `fact_results`, `claim_traces`,
`must_have_results`. Do them in that order — describing a route means quoting
what a specialist FOUND, and doing that first leaves you reaching for prose
when this section asks what MEASURED a number.

SECTION 1 — `fact_results`, one entry per supplied claim

    claim_id       as given
    evidence_ids   every ledger entry you actually read for this claim
    numbers        one entry per MATERIAL mention, including zeros and hedges
    relations      for every threshold, comparison or ranking claim
    verdict        supported | contradicted | unverifiable
    reason         one sentence

Each `numbers` entry: written_value, evidence_id, json_path, trace_kind
(direct | derived), tolerance, and for derived numbers a `calculation` of
{operation, operands} where operation is one of sum, count, max, min, mean,
difference, product, ratio, percent_change, and each operand is
{evidence_id, json_path, select}. Use `select` to name the field to read from
each element when an operand points at a list. Use a derived entry for any
claim about an extreme, total or average: "the peak TSR" must be checked as
max(series), not as whichever bucket happens to hold that value.

A json_path must address a field that EXISTS. Read the entry before pointing
into it and never infer a shape from what would be convenient. Specialist
memory in particular carries two things side by side: a prose `claim` string
summarising the finding, and a `numbers` array holding the measurements it was
drawn from. The measurement lives in the array —

    numbers[0].cdss_score          the value
    claim.CDSS_2024-09             invented; `claim` is a STRING

— and a path into the prose either fails or reads whatever figure happens to
parse out of the sentence, which is not the one you meant.

When a tool's entire result IS the value — a scalar such as "count … = 0" —
use "" as the json_path. When one result states several measurements in a
single string ("sum(...) = $174,897.36 (over 1 row(s); 3 total)"), a path can
address the string but not a figure inside it: return `"unmappable": true`
with a reason rather than pointing at the string and letting the wrong number
be read.

Each `relations` entry is {left, operator, right, reason} where operator is one
of <, <=, >, >=, ==, != and BOTH sides are {evidence_id, json_path} resolving
to real values — including the threshold. Never supply a threshold from memory;
it is usually a field of the same result the measurement came from, such as
`summary.threshold.value`.

Numbers the claim quotes in order to DISOWN them are not checked. Verify the
figure the answer stands behind.

SECTION 2 — `claim_traces`, one entry per supplied claim

Reconstruct the ROUTE that produced the claim. A route has hops, and the shape
that matters is claim -> specialist -> the operations or memory it rested on:

    claim -> general_specialist    synthesis of spend_payments + modeling
          -> spend_payments        summarize_by_group(spends.Amount,
                                   by Merchant Name, sum, top 10)
          -> modeling              memory topic `modeling_Spend Amount_trend`

    claim -> modeling              score_driver_values(month='2025-05')
                                   + memory topic `spend_tsr_cdss_spike_analysis`

Return {claim_id, route, derivation}. `route` is a LIST of hops, each:

    specialist        whose branch this hop is, from the team
    kind              synthesis | operation | memory
    from_specialists  for a synthesis hop, whose findings it combined
    call_ids          for an operation hop, the calls it rests on
    memory_topics     for a memory hop, the KB topics it read
    operations        what those calls actually did, in words: name the table,
                      the aggregation, the grouping, the filter and the window
                      — "summed spends.Amount by Merchant Name over 2025-05",
                      not "queried the data"
    note              one sentence on what this hop contributed

`derivation` is one or two sentences tracing the whole chain.

A route may have ONE hop or several, and may branch: a claim can rest on a
table operation in one specialist AND a memory topic in another. List every
hop that carried it.

The operations and the window are the part a later pass rules on, so write
them precisely. "Analyzed the trend" cannot be judged eligible or ineligible;
"summed spends.Amount by month over 2024-01..2025-06" can.

Use only call ids that appear in the trace, memory topics that appear in the
ledger, and specialists that appear in the team. Every one is checked against
the run, and an invented hop is discarded — the description then loses whatever
rested on it. When a claim rests on NOTHING recorded — the answer asserts it,
but no operation and no memory topic produced it — return an empty `route` and
say so. That is a real finding, not a gap to paper over.

SECTION 3 — `must_have_results`, one entry per supplied baseline point

    must_have_id, verdict (full | partial | miss | not_applicable),
    answer_spans, evidence_ids, reason

A topic mention without satisfying the description is not `full`. Return an
empty list if no baseline was supplied.

TWO RULES THAT DECIDE MORE THAN ANYTHING ABOVE

CITE WHAT MEASURED IT, NOT WHO SUMMARISED IT. The ledger holds tool results and
specialist memory — things that ran an operation — alongside `agent_result`
prose, which is a specialist restating what it found. Only the former is
provenance. An `agent:` id as a claim's ONLY evidence scores that claim
ungrounded however true it is, because a summary is not a measurement. Find the
tool result or memory entry the summary is summarising and cite that; cite the
prose too when it is what the answer echoed, never alone.

A ZERO IS A MEASUREMENT. "no returned payments", "delinquencies were 0" — a
call counted them and got 0, so link it. An unmapped zero is scored as an
invented number, which turns a true negative finding into a hallucination.

Every id you cite is re-checked against the run and every figure against the
payload you said it came from. One that does not resolve is discarded and the
finding resting on it is lost, so cite what you actually read and leave a field
empty when you have nothing to put in it. Return JSON only."""


ELIGIBILITY_PROMPT = """You judge whether the route that produced each claim is
ELIGIBLE for the QUESTION THE USER ASKED. The routes are given to you already
reconstructed; take them as accurate and rule on them.

The brief is part of the route, not the standard. A specialist can execute its
brief perfectly while the brief itself dropped what the question was about —
asked "how did TSR react during the spend spike", an orchestrator that briefed
"describe TSR month by month over the whole window" has already lost the
question, and every claim down that branch is ineligible however faithfully it
was carried out. Judge brief AND operations against the user's question.

Ineligible means the route answers a DIFFERENT question:

  * the WINDOW is wrong — reporting April 2024 when the episode under
    discussion is May 2025. Do not accept the answer's or the brief's choice of
    period; check it against the question and what earlier turns established.
  * the POPULATION or GRAIN is wrong — all cards where commercial was asked, a
    total where per-month was asked.

Return UNAVAILABLE, never NO, when no recorded operation produced the claim.
"We cannot tell" and "this answers the wrong question" are different findings
and only one of them is about the answer.

Return JSON with key `eligibility`: one entry per claim with claim_id, verdict
(YES, NO or UNAVAILABLE), and a one-sentence reason naming what made the route
fit the question or miss it.

TWO RULES, REPEATED LAST. THEY ARE NOT IN TENSION — READ BOTH.

RULE 1 — WHAT MAKES A ROUTE INELIGIBLE IS THE WRONG *SUBJECT*.

Wrong WINDOW, wrong POPULATION, wrong GRAIN.

FIRST, ESTABLISH WHAT THE QUESTION IS ABOUT. A window comes from the question
itself ("in 2025/2026") or from what an earlier turn established ("the recent
reaction" — the reaction the previous answer identified). Say which it is
before you rule.

  * The question or an earlier turn PINS a window, and the route reports a
    different one -> NO. Asked "how did TSR react?" when the previous turn
    established the episode as the May 2025 spike, a branch reporting
    April-September 2024 has answered a different question — every claim down
    it is NO, however cleanly its figures trace and however faithfully the
    specialist executed what it was told. Never take the answer's or the
    brief's choice of period as the question's when the question has one of
    its own.

  * NOTHING pins a window — the question is open and no earlier turn settled
    it -> choosing one is the ANSWER'S JOB, not an error. "In this case, how
    did TSR and CDSS react?" names no period, so a route that identifies a
    defensible episode and reports it is YES. You may not rule NO on the
    grounds that the window was "undefined", "unspecified", or "chosen by the
    route" — that is the analysis being done, and a separate must-have
    requires the answer to commit to a period rather than recite the whole
    range. If you genuinely cannot tell whether the episode chosen is the one
    the case is about, return UNAVAILABLE, never NO.

  * The question NAMES A REFERENCE CONDITION and asks about something else in
    relation to it: "While the TSR and CDSS were reacting in 2025/2026, how was
    the bureau profile?" The reference (the TSR/CDSS reaction) is PART OF THE
    QUESTION, not a rival subject. A claim that establishes it, or that states
    how the asked-about subject stands against it, is ON subject -> YES.

      "TSR peaked at 26.4 in May 2025 (exceeding the risky threshold of 20),
      but this internal signal was not accompanied by adverse bureau trends."

    is YES: it names the reaction the question refers to and reports the
    bureau profile against it. That contrast IS the answer. Ruling it NO
    because "the claim is about TSR, not the bureau profile" mistakes the
    question's own premise for a different question.

    Still NO: a claim that stays WHOLLY on the reference and never reaches the
    asked-about subject. "TSR peaked at 26.4 in May 2025." on its own restates
    the premise and answers nothing about the bureau profile.

A reason of the form "this covers the relevant window" is only valid if you
have said WHICH window the question is about and shown the route matches it.
"The relevant window (2024-05)" is not a finding when the episode is 2025-05;
it is the answer's error restated as its justification. Equally, "the window
is unspecified" is not a finding when the question specified none.

RULE 2 — WHAT DOES *NOT* MAKE A ROUTE INELIGIBLE IS COVERING ONLY PART.

A claim is ATOMIC. It states ONE fact, and covering one merchant, one month or
one metric is what an atomic claim is FOR. If your reason reduces to any of
these, the verdict is YES:

    "X alone does not address the whole question"
    "aggregating by Y does not by itself establish Z"
    "a figure for a single merchant / entity does not answer the broader
     question"

Asked "what transactions are connected", summing spend by merchant IS how
connectedness is found, and each per-merchant total is eligible evidence.
Whether the answer covered the whole question is measured separately; marking
each part ineligible for being a part scores one gap as many.

HOW THEY COMBINE. Rule 2 excuses a claim for being NARROW. It never excuses a
claim for being about the WRONG THING. "TSR was 25.5 in May 2024" is atomic —
and still NO when the episode under discussion is May 2025, because the defect
is its subject, not its scope. Apply rule 1 first; reach for rule 2 only once
you are satisfied the route is on the right window, population and grain."""


MEMORY_LEVERAGE_PROMPT = """You judge whether the system USED the memory it was
given. One call, covering every source at once.

Memory reaching the context window is not the same as memory being used, and
the two come apart in both directions. A system can be handed a knowledge point
and re-query the same table anyway; it can be shown five prior turns and still
ask the user's follow-up as though it were a fresh question. Presence is
already known deterministically — you are asked the other question.

Not leveraging memory is NOT a failure to be penalised, and you should not
grade generously to avoid one. It is a fact worth recording accurately:
leveraged memory is what makes a session cheaper than its turns and what lets
knowledge accumulate across them, so an honest count of where it was and was
not used is the whole point of this pass.

You are given, per source, what was OFFERED this turn:

    kb_topics        knowledge points carried from earlier turns, by topic
    episodic_turns   prior turns of this session, by turn_id and question
    qa_cache         a cached answer replayed for this question, if any

and what the system PRODUCED:

    team             the specialists the orchestrator assembled
    subqueries       the brief it gave each — this is TEAM CONSTRUCTION
    findings         what each specialist returned
    answer           the final answer

Return JSON with key `memory_leverage`: one entry per source that was offered.

    source        kb | episodic | qa_cache
    leveraged     yes | no
    where         construction | specialist_output | answer   (list, may be several)
    items         the exact kb topic strings or episodic turn_ids relied on
    reason        one sentence naming what in the output shows it

WHAT COUNTS AS LEVERAGE

  construction       the brief names a subject, window, entity or threshold
                     that came from a knowledge point or a prior turn rather
                     than from the question. "Analyse TSR around the May 2025
                     spike" leverages a prior turn when the question says only
                     "how did TSR react"; "analyse TSR 2024-01 to 2025-06" does
                     not — that is the full range, which needs no memory.

  specialist_output  a finding states or builds on a remembered value instead
                     of re-deriving it, or explicitly declines to re-query
                     because the KP already answers it.

  answer             the answer resolves a reference ("it", "the reaction",
                     "these cards") using a prior turn, or states a remembered
                     figure and attributes it.

WHAT DOES NOT COUNT

Restating the question. Reaching the same conclusion a prior turn reached, by
re-doing the work — that is convergence, not recall, and re-querying what
memory already held is exactly the inefficiency this measures. Mentioning that
memory exists ("per earlier analysis") without using anything from it.

Every topic and turn_id you cite is checked against what was actually offered;
one that was not shown is discarded. Cite only what you can point to in the
output, and return `leveraged: no` freely — a source offered and unused is a
real and unremarkable result. Return JSON only."""
