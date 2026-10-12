# AI-Generated Text Detector

Classifies text as **AI-generated (1)** or **human-written (0)** using TF-IDF n-grams plus 10 stylometric features, with a Streamlit web app for inference.

**Authors:** Bushra Malik & Ishrat Akbar
**Dataset:** [bushraasaleem/AI-detection-datsets](https://huggingface.co/datasets/bushraasaleem/AI-detection-datsets)

## Results

| Model | Accuracy | F1-Macro |
|---|---|---|
| Logistic Regression (CV, C = 1291.55) | 92.29% | 0.9228 |
| Calibrated LinearSVC (C = 1.0) | 92.45% | 0.9245 |
| **LR + SVC soft-voting ensemble** | **92.67%** | **0.9267** |

Test set: 14,258 samples (80/20 stratified split on source x label). 5-fold CV F1-macro: **0.9139 ± 0.0011**.

**Classification report (ensemble)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Human (0) | 0.92 | 0.94 | 0.93 | 7,116 |
| AI (1) | 0.93 | 0.92 | 0.93 | 7,142 |

**Confusion matrix:** TN 6,655 | FP 461 | FN 584 | TP 6,558

**Per-source accuracy**

| Source | N | Accuracy | FP | FN |
|---|---|---|---|---|
| AI 7k | 1,330 | 99.77% | 0 | 3 |
| AG News | 574 | 98.26% | 10 | 0 |
| OpenWebText | 2,000 | 97.85% | 43 | 0 |
| HC3 | 2,837 | 94.89% | 81 | 64 |
| ArXiv Human | 1,000 | 93.10% | 69 | 0 |
| OpenHermes | 911 | 91.99% | 0 | 73 |
| M4 Human | 1,317 | 88.38% | 153 | 0 |
| M4 AI | 2,717 | 87.52% | 0 | 339 |
| OpenOrca | 820 | 87.20% | 0 | 105 |
| WikiText | 736 | 85.73% | 105 | 0 |
| ELI5 (Reddit) | 16 | 100.00% | 0 | 0 |

Formal academic and encyclopaedic text (M4, WikiText) is hardest, since human and AI styles overlap most there.

## Dataset

71,289 documents after cleaning, deduplication, and class balancing, drawn from 12 sources:

| Source | Label | Domain | Samples |
|---|---|---|---|
| HC3 | Human | Reddit-style Q&A | 7,500 |
| HC3 | AI | ChatGPT replies | 7,500 |
| M4 (RAID) | Human | Academic abstracts (ArXiv, PeerRead, Wikipedia) | 6,584 |
| M4 (RAID) | AI | Davinci / Dolly / Cohere / Llama | 14,914 |
| WikiText-103 | Human | Encyclopaedic text | 3,678 |
| OpenWebText | Human | Web articles | 10,000 |
| OpenHermes | AI | Instruct-model chat responses | 5,000 |
| OpenOrca | AI | GPT-4 / GPT-3.5 instructions | 4,500 |
| AI 7k | AI | GPT-4o / Mistral / Qwen | 7,303 |
| Reddit ELI5 | Human | Casual explanations (few samples recovered) | small |
| AG News | Human | News articles | 2,873 |
| ArXiv Human | Human | Scientific abstracts (bias counterweight) | 5,000 |

**Pre-processing:** unwrap serialised list strings in HC3, lowercase, drop texts under 20 tokens, remove duplicates, balance classes with a source-proportional stratified sampler.

## Features

**TF-IDF:** 5,000 unigrams + bigrams (`min_df=5`, `max_df=0.85`, `sublinear_tf=True`), fit on the training set only.

**Stylometric (10, via spaCy sentencizer, aggregated per document):**

| Feature | Human mean | AI mean |
|---|---|---|
| `chat_word_ratio` | 0.0023 | 0.0018 |
| `punct_ratio` | 0.1339 | 0.1394 |
| `ttr` | 0.8958 | 0.8738 |
| `function_word_ratio` | 0.3835 | 0.3742 |
| `discourse_ratio` | 0.0364 | 0.0392 |
| `sentence_length_std` | 12.97 | 12.51 |
| `sentence_length_cv` | 0.4692 | 0.4425 |
| `contraction_ratio` | 0.0073 | 0.0062 |
| `transition_ratio` | 0.0012 | 0.0022 |
| `formal_start_ratio` | 0.2678 | 0.3364 |

`transition_ratio`, `formal_start_ratio`, and `contraction_ratio` separate the classes best. Stylometric features are scaled (`StandardScaler(with_mean=False)`) and stacked with TF-IDF into a (57,031 x 5,010) training matrix.

## How it works

1. Clean and lowercase the text, split into sentences.
2. Compute and scale the 10 stylometric features.
3. Transform text with the saved TF-IDF vectoriser and stack both feature sets.
4. Score with the ensemble to get `P(AI)` and `P(Human)`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app also needs the trained model files (`best_model.pkl`, `tfidf_vectorizer.pkl`, `feature_scaler.pkl`, `feature_columns.pkl`) and the word lists in this repo.

> Pickle files are sensitive to Python, scikit-learn, and numpy versions. If loading fails, retrain in your own environment.

## Repo structure

```
app.py                 Streamlit app
chat_words.txt         Informal vocabulary list
discourse_markers.txt  Discourse marker list
function_words.txt     Function word list
requirements.txt       Python dependencies (incl. spaCy model)
packages.txt           System packages for deployment
.streamlit/            Server config (max upload size)
```

## Challenges and fixes

- **GPT-3-era bias:** HC3-only training failed on modern models; fixed by adding AI 7k, OpenHermes, and OpenOrca.
- **Academic false positives:** formal human text was flagged as AI; fixed with 5,000 ArXiv human abstracts plus `transition_ratio` and `formal_start_ratio`.
- **Environment mismatch:** pickles broke across Python/scikit-learn versions; fixed with explicit checkpoints.
- **HC3 list strings:** brackets polluted the vocabulary; fixed with `clean_text()` using `ast.literal_eval`.
- **ELI5 deprecation:** the loading script was removed, so OpenWebText replaced it.

## Limitations

- Can be evaded by prompting an LLM to use contractions, vary sentence length, and avoid transitions.
- English only; texts under 20 words are not supported.
- Weakest on formal academic text; probabilities near 0.5 are uncertain.
- Not suitable as sole evidence in academic-integrity decisions.

## Future work

- Add perplexity features (GPT-2) or a fine-tuned RoBERTa
- Add more RAID academic AI sources
- Calibrate probabilities (Platt scaling or isotonic regression)
- Test robustness against paraphrasing and obfuscation

## Reference

Guo, B. et al. (2023). *How Close is ChatGPT to Human Experts?* arXiv:2301.07597 (HC3 dataset).
