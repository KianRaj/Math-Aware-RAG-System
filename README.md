# Math Aware RAG Chatbot

## Audience-Adaptive, Math-Aware Research Assistant

This project implements a Retrieval-Augmented Generation (RAG) system designed to ingest research papers (PDFs) and generate audience-adaptive content, such as presentation scripts, slide decks, and summaries. It features a specialized pipeline for preserving mathematical notation (LaTeX) to prevent hallucinations common in standard LLM ingestion processes.

## System Architecture

The pipeline follows a modular RAG design optimized for scientific papers:

1.  **Ingestion Layer:**
    * **Input:** PDF/LaTeX Papers.
    * **Processing:** `PyMuPDF` for text + Custom Regex for Math Cleaning.
    * **Chunking:** Semantic chunking via `Spacy` (sentence-level).

2.  **Retrieval Layer (Dense):**
    * **Embeddings:** OpenAI `text-embedding-3-small` (High semantic fidelity).
    * **Vector Store:** `ChromaDB` (Persistent local storage).
    * **Filtering:** Metadata filters for Math-heavy chunks.

3.  **Generation Layer:**
    * **Model:** `Google Gemma-7b-it` (Quantized 4-bit).
    * **Orchestration:** HuggingFace Transformers library.
    * **Prompting:** Chain-of-thought with One-Shot examples for strict formatting.

---
## Pipeline Overview

<img width="1020" height="418" alt="pipeline" src="https://github.com/user-attachments/assets/ffe41b5f-3c15-41de-b70c-4c71d11f5277" />



## Key Features

* **Math-Aware Ingestion:** Utilizes custom regex and cleaning pipelines to detect and preserve mathematical formulas, equations, and Greek letters during text extraction.
* **Audience Adaptation:** Dynamically adjusts the output complexity based on target audiences (e.g., General Public, Graduate Students, Policymakers).
* **Iterative Refinement:** Supports conversational memory, allowing users to request specific edits to previously generated slides (e.g., "Make slide 2 less technical").
* **Hybrid Architecture:**
    * **Embeddings:** OpenAI `text-embedding-3-small` for high-quality semantic search.
    * **Vector Store:** ChromaDB for persistent storage and retrieval.
    * **Generation:** Google's `gemma-7b-it` (via Hugging Face Transformers) for local text generation.
* **Citation & Provenance:** Every generated claim is cited with the source page number from the PDF.

## Prerequisites

* Python 3.8 or higher
* CUDA-enabled GPU (Recommended for Gemma-7b inference)
* OpenAI API Key
* Hugging Face Access Token (for Gemma-7b-it model access)

## Installation

1.  Clone the repository.

2.  Install the required Python packages:

    ```bash
    pip install openai torch chromadb pandas numpy requests tqdm spacy transformers python-dotenv pymupdf bitsandbytes accelerate
    ```

3.  Download the Spacy English language model:

    ```bash
    python -m spacy download en_core_web_sm
    ```

## Configuration

1.  Create a file named `.env` in the root directory.
2.  Add your OpenAI API key to the file:

    ```text
    OPENAI_API_KEY=your_sk_key_here
    ```

3.  Ensure you are logged into Hugging Face CLI to access the gated Gemma model:

    ```bash
    huggingface-cli login
    ```

## Project Structure

* **saral.py**: The main script containing the full RAG pipeline (Ingestion, Embedding, Retrieval, Generation).
* **saral_state.py**: Module for managing the active collection state (not included in snippet but imported).
* **Api_key.env**: Environment variable configuration file.
* **saral_chroma_store/**: Local directory created to persist vector embeddings.
* **text_chunks_and_embeddings_df.csv**: Intermediate storage for processed text chunks.

## Usage

### Running the CLI

You can run the script directly to enter the interactive command-line interface:

```bash
python saral.py
```
## Key Features & Implementation

### 1. Math-Aware RAG (Core Innovation)
Unlike standard RAG pipelines that strip special characters, SARAL is designed to preserve the integrity of scientific notation.
* **LaTeX Preservation:** The ingestion pipeline detects and wraps equations in `$$...$$` for proper rendering.
* **Symbol Restoration:** Custom regex cleaners repair common PDF extraction errors (e.g., fixing `x1:T` → `x_{1:T}`, restoring Greek letters).
* **Equation Prioritization:** The retrieval layer prioritizes chunks containing dense mathematical formulas when the user asks for derivations.

![Math Aware RAG Demo]<img width="1644" height="441" alt="Screenshot 2025-11-20 015721" src="https://github.com/user-attachments/assets/e8718c21-0739-4ca1-bd7d-6997890bb9c6" />

*> **Figure 1:** Comparison of standard text extraction vs. SARAL's math-aware cleaning pipeline. Note how the ELBO equation is preserved.*

---

### 2. Audience-Adaptive Generation
The system generates content tailored to specific constraints, supporting multiple lengths and styles via strict system prompting.

| Dimension | Supported Options | Implementation Details |
| :--- | :--- | :--- |
| **Duration** | 30s, 90s, 5min | Controls token count and slide density. |
| **Style** | Technical, Plain-English, Press Release | Adjusts vocabulary complexity and sentence structure. |
| **Format** | Slides, Script, Speaker Notes | Generates structured JSON-like output for easy parsing. |

### 3. Provenance & Citations
To ensure factuality and minimize hallucinations, every generated claim is strictly anchored to the source text.
* **Citation Enforced:** The model is instructed to append `[Page X]` to every bullet point.
* **Source Retrieval:** The UI displays the raw chunks used to generate the answer for verification.

### 4. Iterative Refinement & Change Tracking
The chatbot supports conversational memory for refining outputs.
* **User Query:** *"Make slide #2 less technical."*
* **System Action:** Retrieves the previous answer, isolates Slide 2, and rewrites it while keeping the rest of the deck intact.
* **Delta Tracking:** The system can highlight what changed between versions (e.g., *Changed "Stochastic Differential Equation" to "Random process changes"*).

---

## Evaluation & Metrics

The system was evaluated using a combination of automatic metrics and human review across 3 test papers.

### Automatic Metrics
* **Factuality Proxy:** Semantic overlap between generated claims and source sentences.
* **Citation Coverage:** Percentage of generated sentences containing valid `[Page X]` citations.
* **Quality:** ROUGE-L and BERTScore F1 comparisons against human-authored summaries.

### Human Evaluation
Rated by 3 distinct raters on a 1-5 Likert scale.

| Metric | Score (Avg) | Description |
| :--- | :---: | :--- |
| **Audience Appropriateness** | **4/5** | Did the output match the requested style (Technical vs Plain)? |
| **Factuality** | **3/5** | Accuracy of the math and claims derived from the paper. |
| **Helpfulness** | **4/5** | Utility of the generated scripts for a real presentation. |

> **Note:** Full evaluation logs and ROUGE scores can be found in the `evaluation/reports/` directory.
### EVALUATION REPORT AGGREGATED RESULTS
```bash
python evaluate.py
```
### 📊 Evaluation Execution Log

```yaml
--- Running Evaluation for P1_VDM ---
Automatic Metrics:
  Semantic_Overlap_Factuality_Proxy_OpenAI: 0.0
  Citation_Coverage: 0.3333
Quality Metrics:
  ROUGE_rouge1_F1: 0.2679
  ROUGE_rouge2_F1: 0.0483
  ROUGE_rougeL_F1: 0.1914
  BERTScore_F1:    0.8113

--- Running Evaluation for P2_MFP3D ---
Automatic Metrics:
  Semantic_Overlap_Factuality_Proxy_OpenAI: 0.0
  Citation_Coverage: 0.1429
Quality Metrics:
  ROUGE_rouge1_F1: 0.3203
  ROUGE_rouge2_F1: 0.0394
  ROUGE_rougeL_F1: 0.1719
  BERTScore_F1:    0.8482

--- Running Evaluation for P3_MetaFood3D ---
Automatic Metrics:
  Semantic_Overlap_Factuality_Proxy_OpenAI: 0.0
  Citation_Coverage: 0.1667
Quality Metrics:
  ROUGE_rouge1_F1: 0.3547
  ROUGE_rouge2_F1: 0.1369
  ROUGE_rougeL_F1: 0.1962
  BERTScore_F1:    0.8299

```bash
python result.py ## Run all the Evaluation folder python file one by one. last one result would look like 
```


| Paper_ID | Citation_Coverage | ROUGE_rougeL_F1 | BERTScore_F1 | Audience_Appropriateness_Rating | Factuality_Rating | Helpfulness_Rating | FINAL_COMPOSITE_SCORE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P1_VDM | 0.3333 | 0.1860 | 0.8091 | 4.666667 | 4.666667 | 4.333333 | 79.49 |
| P2_MFP3D | 0.1429 | 0.1719 | 0.8482 | 3.333333 | 3.666667 | 3.333333 | 69.29 |
| P3_MetaFood3D | 0.1667 | 0.1962 | 0.8299 | 4.666667 | 5.000000 | 4.666667 | 83.37 |

---
**SYSTEM-WIDE AVERAGE SCORE: 77.38 / 100**

