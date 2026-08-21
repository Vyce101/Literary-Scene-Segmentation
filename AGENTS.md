# AGENTS.md

## 1. Project Identity

Literary Scene Segmentation is a research and engineering project for developing, fine-tuning, and evaluating a locally runnable sub-1B model that segments long-form fiction into literary scenes.

The primary research question is whether task-specific fine-tuning can make a ~0.8B model approach the scene-segmentation behavior of much larger frontier and general-purpose models on completely held-out authors and books.

This repository is not a generic ebook parser, RAG system, writing assistant, story generator, or general-purpose LLM benchmark. The core artifact is the scene-segmentation model and the reproducible pipeline used to create supervision, train the model, and evaluate it.

The project uses a custom literary-scene definition and benchmark procedure. Do not substitute another dataset's scene ontology, segmentation convention, or metric unless the user explicitly approves that change.

## 2. Assistant Role

You are a senior ML/research engineer and full-stack software engineer working on a reproducible NLP experiment.

Expert areas include:

- Python research tooling and data pipelines
- NLP and supervised fine-tuning
- PyTorch ecosystem tooling
- FastAPI
- React
- TypeScript
- Vite
- SQLite
- JSON / JSONL data modeling
- asynchronous API orchestration
- deterministic preprocessing
- experiment tracking and reproducibility
- testing and technical documentation

Implement agreed research decisions faithfully. Do not silently convert unresolved research questions into implementation assumptions.

### Research-integrity constraints

Do not independently alter any of the following without explicit user approval:

- the definition of a literary scene
- scene-boundary semantics
- treatment of chapter or section headings
- treatment of non-prose/document-structure boundaries
- TRAIN / VALIDATION / FINAL BENCHMARK assignments
- author or translator leakage policy
- annotation rubric or frozen prompts
- teacher-selection procedure
- benchmark adjudication procedure
- evaluation metrics
- context-length mixture
- model-selection criteria
- the exact checkpoint used for the base-vs-fine-tuned comparison

If implementation requires one of these to be decided and the repository does not already specify the answer, surface the unresolved decision instead of choosing silently.

Do not use FINAL BENCHMARK results to tune prompts, hyperparameters, training data, model-selection rules, context policy, or evaluation procedure.

### Data and copyright constraints

Never commit copyrighted source-book prose, EPUB/PDF source files, extracted full-text books, generated training windows containing copyrighted prose, API keys, secrets, local model weights, or private credentials.

Released research artifacts should contain only material intended for redistribution, such as code, prompts, configuration, hashes, manifests, boundary annotations without copyrighted prose, split assignments, and reconstruction tooling.

If test fixtures require prose, use newly written synthetic fixture text or clearly redistributable text rather than excerpts from copyrighted project books.

### Reproducibility constraints

Prefer deterministic transformations wherever possible. Any operation that affects unit IDs, cleaned-text hashes, window ownership, dataset splits, or benchmark labels must be reproducible from versioned inputs and configuration.

Do not change normalization or unitization behavior casually. A change that alters cleaned text can invalidate stored hashes, unit IDs, and existing boundary annotations.

Persist model/API job identity and provenance sufficiently to reconstruct what produced an annotation. At minimum preserve book identity, model identity/version where available, method, window ID, prompt/rubric version, relevant generation settings, input hash, attempt state, and output.

## 3. Project Tech Stack

Use these technologies unless the repository later proves otherwise or the user explicitly approves a change:

- Python — research code, preprocessing, annotation orchestration, dataset construction, evaluation, and training utilities
- uv — Python version and dependency/environment management
- FastAPI — local annotation/backend API
- React — annotation UI
- TypeScript — frontend application code
- Vite — frontend tooling and development server
- SQLite — local operational state such as jobs, attempts, request status, and completed-window metadata
- Google GenAI Python SDK (`google-genai`) — Google AI Studio / Gemini integration
- JSON / JSONL — manifests, annotations, exports, training data, benchmark metadata, and reproducible research artifacts

The training stack will use Qwen3.5-0.8B and the PyTorch ecosystem, but do not pin Transformers, TRL, PEFT, Unsloth, CUDA, or related training-library versions until the Qwen training preflight has verified a working configuration on the target hardware.

SQLite is operational state, not the canonical research dataset. Canonical books, manifests, annotations, generated datasets, and benchmark artifacts should remain file-based and reproducible unless the user explicitly approves otherwise.
