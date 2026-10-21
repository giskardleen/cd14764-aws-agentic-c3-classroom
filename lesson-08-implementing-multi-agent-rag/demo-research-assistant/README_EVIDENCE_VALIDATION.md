# Evidence Sufficiency Validation — Multi-Agent RAG Experiment

> **Lesson 8 · Experimental extension**
> **Baseline:** `research_assistant_rag.py`
> **Experimental implementation:** `research_assistant_rag_validated.py`
> **Scope:** Evaluate whether retrieved passages *support the requested answer*, rather than treating retrieval scores as proof of factual relevance.

## 1. Purpose and hypothesis

The Lesson 8 multi-agent RAG demo retrieves passages from two Amazon Bedrock Knowledge Bases (Computer Science and Biology), aggregates and ranks the results, and sends the selected passages to a synthesis agent.

During baseline testing, an out-of-scope question about blockchain consensus for IoT networks retrieved ten passages and had the **highest observed mean Top-K retrieval score (0.726)** among the three original scenarios. The baseline synthesis agent nevertheless recognized that the documents did not support an answer and declined to invent one.

This motivated a separate experiment:

**Can an explicit evidence-sufficiency gate distinguish retrieval similarity from answer support, select only supporting passages, and control whether synthesis runs?**

The hypothesis is not that an LLM validator proves factual truth. It is that making the support decision explicit creates an observable, testable orchestration boundary.

## 2. Files and separation from the baseline

```text
demo-research-assistant/
├── research_assistant_rag.py             # Course demo / baseline; unchanged by experiment
├── research_assistant_rag_validated.py   # Evidence-sufficiency extension
└── README_EVIDENCE_VALIDATION.md         # This experimental record
```

The experimental module imports the baseline implementation and reuses its retriever builders, retry helper, aggregation, model configuration, and synthesis agent. It does **not** replace or edit the baseline file. The experimental gate is implemented in the separate Python module.

### Baseline architecture

```text
                              Research question
                                      |
                         +------------+------------+
                         |                         |
                  CS Retriever              Bio Retriever
                         |                         |
                         +------------+------------+
                                      |
                            Aggregate / rank Top-K
                                      |
                               Synthesis Agent
                                      |
                          Grounded answer or refusal
```

The baseline synthesis agent receives the ranked passages and is instructed to use only retrieved information and acknowledge missing evidence. There is **no separate pre-synthesis sufficiency decision**.

### Experimental architecture

```text
                              Research question
                                      |
                         +------------+------------+
                         |                         |
                  CS Retriever              Bio Retriever
                         |                         |
                         +------------+------------+
                                      |
                            Aggregate / rank Top-K
                                      |
                             Evidence Validator
                         (Nova Pro / Bedrock Converse)
                                      |
                        Validate JSON + document IDs
                                      |
                +---------------------+---------------------+
                |                     |                     |
           SUFFICIENT              PARTIAL             INSUFFICIENT
                |                     |                     |
          Select supporting     Select supporting      No supporting
              passages              passages             passages
                |                     |                     |
                +----------+----------+                     |
                           |                                |
                    Synthesis Agent                    Skip synthesis
                           |                                |
                    Grounded answer                    Abstention

          Invalid output / validator failure -> VALIDATION_ERROR
          Both retrievers fail               -> RETRIEVAL_ERROR
          Synthesis failure                  -> SYNTHESIS_ERROR
```

The validator is a **model-based semantic assessment**. The gate that checks the returned schema, document IDs, and allowed transitions is **programmatic**. A valid JSON decision is not proof that the semantic judgment is correct.

## 3. Validation contract and control flow

The validator receives the research question and the selected Top-K passages (IDs, titles, and contents). It does **not** receive retrieval similarity scores and is instructed not to use outside knowledge.

Expected response shape:

```json
{
  "status": "PARTIAL",
  "supporting_doc_ids": ["Biology Papers-1"],
  "reason": "Evidence supports wheat, but not rice."
}
```

| Status | Meaning | Orchestration action |
| --- | --- | --- |
| `SUFFICIENT` | Passages directly support a substantive answer to the question as asked | Synthesize using only selected supporting passages |
| `PARTIAL` | Passages support only a limited portion of the requested scope | Synthesize supported aspects only; disclose missing evidence |
| `INSUFFICIENT` | Passages do not support a useful answer | Skip synthesis and abstain |
| `VALIDATION_ERROR` | Validator invocation or output-contract validation failed | Report error; do not claim evidence was insufficient |

`parse_validation()` checks that the output is a JSON object, the status is allowed, the reason is nonempty, document IDs are strings without duplicates, every cited ID exists in the supplied passages, and status/ID combinations are consistent. `select_supported_passages()` passes only selected documents to synthesis.

Additional operational outcomes are `RETRIEVAL_ERROR` (both retrievers failed) and `SYNTHESIS_ERROR` (synthesis raised an exception). If one retriever fails but the other succeeds, the experiment continues with available evidence and records the retriever error.

## 4. Test environment and methodology

- **Platform:** Amazon Bedrock, region `us-east-1`.
- **Knowledge Bases:** two course/demo KBs, one for Computer Science and one for Biology, populated with six example documents each.
- **Retrieval:** parallel Strands retriever agents; each KB returned five passages in the observed runs.
- **Aggregation:** baseline Top-K logic; five passages selected per query.
- **Synthesis:** existing Nova Pro-based Strands synthesis agent.
- **Validation:** Nova Pro via Bedrock Converse, temperature `0`, requested JSON decision.
- **Scoring:** mean score of the *aggregated Top-K* passages; scores are **not** used as a validation threshold.
- **Evidence:** terminal output from one observed execution per query, plus six local contract tests.

The example documents are lesson fixtures. Their titles, venue names, and numerical claims should not be interpreted as independently verified scientific literature.

## 5. Results: baseline vs. validated experiment

### 5.1 Scenario comparison

| # | Query | Mean Top-K score | Baseline (observed) | Validated (observed) | Supporting passages | Validated action |
| --- | --- | ---: | --- | --- | ---: | --- |
| 1 | `applications of machine learning in genomics` | 0.705 | Synthesized an evidence-based answer | `SUFFICIENT` | 4 / 5 | Synthesized from four selected passages |
| 2 | `CRISPR gene editing for crop improvement` | 0.712 | Synthesized an evidence-based answer | `SUFFICIENT` | 1 / 5 | Synthesized from one selected passage |
| 3 | `blockchain consensus mechanisms for IoT networks` | **0.726** | Synthesis agent declined to answer for lack of supporting documents | `INSUFFICIENT` | 0 / 5 | **Synthesis skipped** |
| 4 | `How does CRISPR improve drought resistance in wheat, and what evidence demonstrates its effectiveness in rice?` | 0.716 | **Not executed on baseline** | `PARTIAL` | 1 / 5 | Synthesized wheat evidence; explicitly noted lack of rice evidence |

**Comparison boundary:** The first three baseline outcomes were observed in an earlier demo run. The fourth query was run **only** against the validated extension. Since the baseline always sends Top-K passages to synthesis, it would invoke synthesis for this fourth query *by design*, but its actual response to that query was **not measured**. The table therefore does not attribute an invented baseline outcome to scenario 4.

### 5.2 Evidence selected and decisions

**Scenario 1 — Genomics (`SUFFICIENT`)**
The validator selected `CS Papers-1`, `Biology Papers-1`, `CS Papers-2`, and `Biology Papers-2`. The synthesis discussed gene expression prediction, cancer biomarkers, drug discovery, and antibiotic-resistance analysis. The validator accepted four of the five Top-K passages.

**Scenario 2 — CRISPR crop improvement (`SUFFICIENT`)**
The validator selected only `Biology Papers-1`. The resulting answer focused on the fixture's CRISPR-Cas9 wheat example, including the stated TaDREB2 knockout, 40% water-use efficiency improvement, and 15–25% yield improvement under water stress. These numbers are claims in the demo corpus, **not independently verified research findings**.

**Scenario 3 — Blockchain / IoT (`INSUFFICIENT`)**
The validator returned no supporting document IDs and explained that the retrieved excerpts did not address blockchain consensus mechanisms for IoT. The orchestrator skipped synthesis; the structured result recorded `synthesis_seconds: null`. This query had the highest observed average retrieval score, illustrating why score ranking alone cannot establish answer support.

**Scenario 4 — Wheat + rice (`PARTIAL`)**
The validator selected `Biology Papers-1`, recognizing that the wheat portion was supported but the requested rice evidence was absent. The synthesis answered the wheat portion and explicitly stated that the retrieved passage contained no evidence about rice. The result was `PARTIAL`, with one supported passage and a Top-K mean score of `0.716`.

### 5.3 Measured timings in the validated runs

These timings are single-run observations, not benchmarks or matched baseline comparisons.

| Scenario | Retrieval | Validation | Synthesis |
| --- | ---: | ---: | ---: |
| Genomics | 5.38 s | 1.47 s | 3.07 s |
| CRISPR crops | 4.64 s | 1.44 s | 1.85 s |
| Blockchain / IoT | 3.92 s | 1.46 s | Not invoked |
| Wheat + rice | 6.17 s | 1.51 s | 2.33 s |

The validation stage adds an additional model call when passages are available. Its latency and cost must be evaluated separately before adopting this design in production.

## 6. Contract tests

Run without AWS model calls:

```bash
python research_assistant_rag_validated.py --self-test
```

Observed result: **6 tests passed**.

| Test | Expected behavior | Result |
| --- | --- | --- |
| `test_sufficient` | Accept supported decision and select the cited passage | PASS |
| `test_partial` | Accept partial decision and select cited passage | PASS |
| `test_insufficient` | Accept insufficient decision and select no passages | PASS |
| `test_unretrieved_id_rejected` | Reject citations to documents outside supplied passages | PASS |
| `test_invalid_json_rejected` | Reject malformed validator output | PASS |
| `test_insufficient_with_citation_rejected` | Reject `INSUFFICIENT` with supporting IDs | PASS |

These tests exercise **schema/transition logic with controlled inputs**, not the accuracy of the LLM's semantic classifications. They do not cover every possible error path.

## 7. Reproduction

From `lesson-08-implementing-multi-agent-rag/demo-research-assistant`, with the lesson's dependencies and `.env` configured:

```bash
# 1. Verify the deterministic output contract (no AWS calls)
python research_assistant_rag_validated.py --self-test

# 2. Run the three built-in experimental scenarios (AWS calls)
python research_assistant_rag_validated.py

# 3. Run the additional partial-evidence scenario (AWS calls)
python - <<'PY'
import research_assistant_rag_validated as experiment

query = {
    "query": (
        "How does CRISPR improve drought resistance in wheat, "
        "and what evidence demonstrates its effectiveness in rice?"
    )
}
result = experiment.run_validated_query(query)
print(result)
PY
```

To rerun the baseline independently:

```bash
python research_assistant_rag.py
```

The baseline demo was previously adjusted to use `managedSearchConfiguration` for Bedrock Managed Knowledge Bases. That compatibility fix is part of the existing baseline, **not** the evidence-validation extension.

## 8. Limitations and interpretation

1. **Semantic validation is probabilistic.** The Nova Pro validator may accept unsupported evidence, reject relevant evidence, or classify a partial question as sufficient. Temperature `0` does not guarantee identical outcomes across runs.
2. **A valid contract is not a factual guarantee.** The program verifies the JSON shape and that referenced IDs were retrieved; it does not independently verify that those passages entail the validator's claims.
3. **Small, synthetic corpus.** Twelve lesson example documents and four queries cannot establish production accuracy or generalizability. The fixture's scientific claims were not independently fact-checked.
4. **No repeated-run evaluation.** Each live scenario was observed once. There are no confidence intervals, precision/recall estimates, adversarial probes, or labeled evaluation dataset.
5. **Not a controlled latency or cost comparison.** The observed timings are for the validated runs; baseline and validated runs were not benchmarked under matched conditions. The extra Converse invocation has a cost.
6. **Retrieval scores are not factual confidence.** The reported average is a ranking-related similarity measure for the retrieved Top-K, not a calibrated probability of correctness or evidence sufficiency.
7. **Top-K bounds the available evidence.** Relevant documents omitted during retrieval or aggregation cannot be rescued by the validator. The gate judges the passages it receives, not the complete knowledge base.
8. **Document-level selection is coarse.** The validator returns passage IDs, not claim-level entailment mappings. A selected passage can contain both relevant and irrelevant statements.
9. **Citation format is not strictly enforced in synthesis.** In the fourth run, the synthesis output used `[DOC_ID: Biology Papers-1]` rather than the baseline prompt's requested `[DOC_ID]` format. The selected ID was valid, but exact formatting is not guaranteed by prompting alone.
10. **Parallel agent logs interleave.** Retriever outputs may appear mixed in terminal logs. Structured orchestration results are more reliable for comparing status and selected IDs.
11. **Operational failure handling is not fully tested.** The code distinguishes validation, retrieval, and synthesis errors, but the six local tests do not exercise all live outage, throttling, malformed-response, or partial-retriever-failure scenarios.
12. **No observed baseline for scenario 4.** Its baseline behavior should not be described as a measured outcome until that query is executed against the baseline implementation.

## 9. Engineering takeaway

```text
Retrieval similarity != evidence sufficiency != answer correctness
```

The baseline synthesis agent already abstained on the out-of-scope blockchain query. The experimental extension does **not** prove the baseline was incorrect or that the new design is more accurate in general. What it demonstrates in these observed cases is a **different allocation of responsibility**:

- The retrievers find candidate passages.
- The validator assesses whether those passages support the question.
- The program checks the decision contract and controls whether synthesis is allowed.
- The synthesis agent answers only from selected supporting passages, or is not invoked.

This explicit gate makes the decision observable and testable, while leaving the underlying semantic assessment fallible. The baseline is retained for direct comparison and further experiments.
