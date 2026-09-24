
# Cited Q&A over Company Filings — Prism trial project

## 1. Problem

Prism's LLM layer (Spectra) is supposed to let users ask natural-language questions about companies and get answers sourced from primary filings and structured data, not a web search — every claim cited back to its exact source.

## 2. What I built

A small pipeline, in scope for ~3-4 hours:

- **`data/filings.json`** — 15 synthetic documents across 4 fictional Indian companies (steel, pharma, textiles, logistics), spanning quarterly results, earnings-call transcripts, insider trading disclosures, shareholding patterns, a guidance-vs-delivery note, an alternative-data note, and a regulatory monitor — deliberately mirroring Prism's stated data categories.

- **`rag.py`** — The core retrieval and synthesis engine.

    - Retrieval: Uses TF-IDF and cosine similarity over a concatenated string of document metadata (company, doc_type, quarter, section) and text.

    - Guardrails: If the question names a fiscal period (e.g., Q1FY27) that contradicts the best-matched chunk, its score is heavily penalized. If the final score falls below a set threshold (THERSOLD), the system abstains.

    - Generation: Calls a hosted DeepSeek-V4.1-Flash model via the Hugging Face API. The system prompt hard-constrains the LLM to use only the provided context and end every sentence with an exact [doc_id] citation tag.

- **`data/eval_set.json`** — 10 hand-written questions: 8 answerable (including one that requires cross-referencing two sources), 2 adversarial (a company that doesn't exist in the corpus; a real company but a time period with no available data).

- **`eval.py`** — An automated test loop that scores citation accuracy and correct abstention handling.

- **`demo.py`** — A simple script for testing single queries interactively.

## Setup

1. Create a Python virtual environment and activate it.
2. Install dependencies: ``` pip install -r requirements.txt ```
3. Create a `.env` file in the project root: 
```
HF_TOKEN=your_huggingface_token
THERSOLD=0.2

```


## Run

```
python3 demo.py
python3 eval.py

```

## 3. Key assumptions

- **Synthetic data.** I didn't scrape or reproduce any real filing text — the corpus is fictional to keep this legally clean and self-contained. The retrieval/synthesis logic is agnostic to that; swap in real filings and it behaves the same.
- **Direct LLM API over LangChain.** The generation step utilizes the huggingface_hub InferenceClient rather than LangChain. For a focused pipeline like this (retrieve, prompt, generate), LangChain adds indirection without capability. A direct API call is much more auditable and makes the actual hard problem — strict citation-grounding — easier to see, debug, and unit-test.
- **TF-IDF + Metadata concatenation.** TF-IDF was chosen because it's dependency-light and fully deterministic. However, because financial filings often omit the company name or quarter in the body text (e.g., margins sections), I concatenated the metadata tags into the searchable text space. This ensures causal questions like "Why did Chandra Textiles' margins contract in Q1FY27?" map correctly to the source text.


## 4. What I'd change before production use

1. **Dense retrieval + structural filtering.** Relying on TF-IDF string matching and post-hoc score penalties for date mismatches is a good way for the prototype. A production system should use dense vector embeddings for semantic recall, combined with hard structured database filters (company ID, fiscal period, doc type) applied before the similarity ranking.

2. **A much larger, adversarial eval set.** We need to increace the eval set. Although 10 quiestion is enough to find one isssue but to check it properly we need hundreads of question.

3. **Real cost/latency tracking.** Right now as we are using the free tier we do not have track cost and latecny but using the paid while production it becoms the one of the important tasks. 
