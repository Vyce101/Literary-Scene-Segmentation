# Measurement & Evaluation Contract

Status: **Pre-annotation working contract**

This document records the experimental rules that must be settled before large-scale annotation, training-data generation, or final benchmark evaluation. Once marked frozen, changes require an explicit version change and must not be made in response to FINAL BENCHMARK results.

## 1. Primary Research Question

Can task-specific fine-tuning make a ~0.8B locally runnable language model approach the literary scene-segmentation performance of much larger frontier and general-purpose models on completely held-out authors and books?

The primary model is **Qwen3.5-0.8B**.

The primary claim, if supported, is specialization/compression rather than human-level segmentation.

A defensible successful-result claim is of the form:

> A sub-1B fine-tuned model matches or approaches much larger frontier/general-purpose models on held-out literary scene segmentation while running locally at far lower model size and inference cost.

Do not claim human-level scene segmentation unless a compatible human-annotated evaluation supports that claim.

## 2. Literary Scene Construct

A scene is a contiguous stretch of narrative forming one coherent dramatic or narrative situation.

Narrative unity is the primary criterion. Evidence for a new scene can include meaningful changes in story time, location, point of view/focalization, principal characters, narrative thread, immediate goal, conflict, or dramatic situation.

No single factor automatically creates a new scene. A boundary is created only when the narrative meaningfully leaves one coherent situation and begins another.

The frozen scene rubric and annotation prompt will live separately under a versioned prompt/specification file. The same frozen rubric must be used for training supervision, validation, final benchmark creation, model evaluation, and normal inference unless an experiment explicitly studies a different prompt version.

## 3. Boundary Semantics

A literary scene boundary is represented by the unit ID of the **first unit of the new scene**.

Chapter and section headings attach to the material that follows them. If a genuine new scene begins at a chapter or section transition, the boundary is placed **before the heading**, not between the heading and its following prose.

Both teacher annotations and the fine-tuned model use the same two-list JSON output:

```json
{
  "boundaries_before": [],
  "document_boundaries_before": []
}
```

`boundaries_before` contains all predicted boundary unit IDs. `document_boundaries_before` contains the subset of those IDs that separate meaningful document/non-narrative structure. Every ID in `document_boundaries_before` must also appear in `boundaries_before`. Boundary types are preserved in annotation and evaluation metadata. The output does not contain separator characters. Rendering separators such as `***` or `///` is a deterministic application-layer operation.

## 4. Non-Prose / Document-Structure Boundaries

Front matter and other non-prose structural material may remain present in operational inputs so the deployed system can handle real books.

Document-structure boundaries mark transitions between coherent document/non-narrative material such as copyright/publication information, tables of contents, dedications/prefaces, acknowledgements, appendices, and the narrative itself. Individual lines or items within one coherent section do not each receive a separate document-structure boundary. Ordinary chapter or section headings are not document boundaries merely because they are structural headings.

Document-structure boundaries are a **separate annotation type** from literary scene boundaries, and the type must be preserved in annotation and evaluation metadata even though the application renders both through the same `boundaries_before` output and may use the same visible separator.

The **primary literary-scene metric scores literary scene boundaries separately** and excludes document-structure boundaries from that score. Document-structure handling may be evaluated and reported separately as an operational robustness result.

Do not silently mix document-structure boundaries into the literary-scene metric.

## 5. Source Unitization and Window Ownership

Every sentence, heading, or other standalone structural item receives a deterministic unit ID while paragraph structure is preserved.

For overlapping annotation/inference windows, each request contains:

- PAST: context only
- TARGET: the only region where predictions are allowed
- FUTURE: context only

Every source unit belongs to exactly one TARGET region for a given window configuration.

PAST, TARGET, and FUTURE positions are calculated deterministically from the original unitized book before model requests are sent.

Predictions from earlier windows must never alter the contents of later windows.

Requests may run concurrently and responses may arrive out of order. Results are merged using stable book/model/method/window identity.

## 6. Teacher Selection

The teacher must not be selected from performance on a single novel.

Teacher selection uses a **diverse calibration corpus** spanning meaningfully different literary styles and kept separate from FINAL BENCHMARK material.

Candidate model outputs should be anonymized during qualitative comparison where practical.

GPT-5.6 Sol may act as one critic/judge during teacher selection, but must not be the sole authority, especially if GPT-5.6 Sol is itself a teacher candidate.

Manual inspection and disagreement analysis are part of teacher selection. If a cheaper/cleaner Google model is effectively equivalent to GPT-5.6 Sol under the frozen rubric, prefer the simpler teacher setup. If it is materially worse, GPT-5.6 Sol may be retained as teacher.

The teacher-selection procedure and selected teacher must be frozen before full training-data generation.

## 7. Training / Validation / Final Separation

Split by author before generating training windows.

No author used in FINAL BENCHMARK may appear anywhere in TRAIN or VALIDATION.

No FINAL BENCHMARK result may be used to alter model weights, prompts, training data, hyperparameters, context-length mixture, checkpoint selection, inference policy, adjudication rules, or evaluation metrics.

### TRAIN

Use the approved professional-fiction training corpus. The target is approximately 6 million unique cleaned words before contextual overlap.

TRAIN author assignments remain fixed. After deterministic cleaned word counts are measured, the exact number of volumes used from already-approved TRAIN authors may be adjusted to target approximately 6 million unique cleaned words before contextual overlap. Do not change the benchmark split merely to force TRAIN to equal exactly 6,000,000 words. All NieR material is excluded from TRAIN.

### VALIDATION

Validation includes the two approved fanfiction works:

- *Making Arrangements* — Crowns of Laurels
- *Second Wind* — Quill Q

Validation also includes the professionally published work **_NieR:Automata — Long Story Short_**, whose author is completely held out from TRAIN and FINAL BENCHMARK. *Short Story Long* remains unused/reserve material.

VALIDATION may be used for prompt/inference-policy development, training hyperparameter choices, checkpoint/model selection, decisions about additional training, and failure analysis.

VALIDATION must never be mixed into training.

### FINAL BENCHMARK

FINAL BENCHMARK uses only authors completely absent from TRAIN and VALIDATION.

Current approved final material:

- J. K. Rowling — *Harry Potter and the Philosopher's Stone*, *Chamber of Secrets*, *Prisoner of Azkaban*, *Goblet of Fire*
- George R. R. Martin — *A Game of Thrones*
- SunSunSun — *Alya Sometimes Hides Her Feelings in Russian*, Volumes 1–3

The final benchmark remains sealed until the model, prompt, training procedure, checkpoint-selection rule, and evaluation procedure are fixed. The FINAL BENCHMARK assignments above remain unchanged.

## 8. Translator Leakage Audit

Author holdout alone is not assumed to guarantee prose-style independence for translated fiction.

Book manifests must record, where available:

- original author
- English translator
- publisher / edition
- source language

Before freezing splits, audit whether an English translator appears across TRAIN, VALIDATION, and FINAL BENCHMARK.

Prefer avoiding translator overlap where practical. If complete separation is impractical, record and disclose the overlap rather than claiming that author holdout guarantees translator holdout.

## 9. Benchmark Annotation and Disagreement

The fine-tuned ~0.8B model being evaluated must play no role in FINAL BENCHMARK annotation or adjudication.

Multiple strong annotators independently annotate the held-out benchmark under the exact frozen scene definition and procedure.

Raw individual annotations must be preserved.

Evaluation must distinguish at least:

- placement disagreement: annotators identify the same conceptual transition but choose nearby unit IDs
- existence disagreement: annotators disagree about whether a scene boundary exists
- clearly incorrect predictions

Do not collapse all disagreement into a single exact point before preserving the original annotations.

The exact clustering, adjudication, acceptable-range representation, and matching algorithm must be finalized and frozen before FINAL BENCHMARK model evaluation.

## 10. Evaluation Baselines

At minimum compare under the same held-out books, scene definition, allowed context, and evaluation procedure:

- fine-tuned Qwen3.5-0.8B
- the exact untouched checkpoint from which the fine-tuned model was initialized
- teacher frontier model
- another frontier model
- modern ~4B local general-purpose LLM prompted without task-specific fine-tuning

Additional meaningful baselines may be included if cheap and methodologically compatible.

Published scene-segmentation systems may be discussed as related work. They may only be treated as direct quantitative baselines if they can reasonably be applied to this project's exact task definition and evaluation protocol.

Do not compare scores from incompatible scene ontologies, languages, segmentation units, or metrics as though they were equivalent.

## 11. Metrics

The primary benchmark must not rely on simple positional accuracy because scene boundaries are sparse.

The final metric suite must include measures that expose over-segmentation and under-segmentation and must account for near-placement disagreement without rewarding duplicate predictions around one conceptual boundary.

At minimum, the metric design should include:

- exact boundary precision / recall / F1
- an adjudication-aware or range-aware boundary score
- at least one segmentation-level measure suitable for boundary segmentation

The exact range tolerance, matching procedure, and segmentation metric must be chosen and frozen before FINAL BENCHMARK evaluation.

Human-human or strong-annotator agreement is evidence about task ambiguity; it is not automatically a mathematical upper bound on model performance.

## 12. Training Context Budget

The trainer's maximum serialized sequence length is targeted at 8K tokens, subject to hardware preflight.

Per-example context budgets will be mixed rather than forcing every example to the maximum length.

The nominal 2K / 4K / 8K mixture must be calculated using the **fully serialized Qwen training example**, including:

- chat template
- SYSTEM prompt
- USER structural markup
- PAST / TARGET / FUTURE text
- ASSISTANT JSON target
- required special tokens

The dataset generator must use the exact Qwen tokenizer used for training.

No serialized example may exceed the trainer's true maximum sequence length. The literary-text portion therefore receives only the remaining token budget after fixed and variable formatting/output overhead is accounted for.

## 13. Evaluation Efficiency Measurements

Record alongside segmentation quality:

- parameter count
- inference speed
- memory / VRAM usage
- context length used
- average scene length
- minimum scene length
- maximum scene length
- scene-length distribution
- major observed failure modes

The purpose is to evaluate both segmentation quality and specialization/compression efficiency.

## 14. Reproducibility and Copyright

Do not distribute copyrighted source prose or generated training examples containing copyrighted prose.

Release reconstruction tooling so a researcher with matching source copies can reproduce the processed corpus and training examples.

Released artifacts should include, as appropriate:

- preprocessing and dataset-generation code
- boundary annotations without copyrighted prose
- book / edition manifests
- cleaned-source hashes
- prompt / rubric versions
- split assignments
- dataset-generation configuration

Reconstruction must verify the supplied source against the expected processed version/hash before applying unit-ID-based annotations. Mismatches should warn or fail rather than silently producing misaligned data.

## 15. Freeze Points

Before full TRAIN annotation begins, freeze:

- scene definition and rubric
- heading semantics
- non-prose annotation policy
- unitization rules
- teacher-selection result
- annotation prompt/version
- annotation generation settings

Before training begins, additionally freeze:

- TRAIN / VALIDATION / FINAL split
- translator-overlap audit result
- training serialization format
- context-budget mixture
- initial model checkpoint
- checkpoint-selection procedure

Before FINAL BENCHMARK evaluation begins, additionally freeze:

- benchmark annotations/adjudication
- disagreement/range representation
- metric implementation
- matching algorithm
- inference policy for every compared model

FINAL BENCHMARK is analysis-only after opening. Its results do not feed back into model or procedure development.
