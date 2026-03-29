# Hindi ASR Fine-tuning & Evaluation Pipeline
### Speech AI & NLP Project

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Whisper](https://img.shields.io/badge/OpenAI-Whisper-black)
![MuRIL](https://img.shields.io/badge/Google-MuRIL-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What This Project Is About

This project tackles four real problems that come up when building a Hindi speech recognition (ASR) system. Starting from raw audio, we fine-tune a Whisper model on real Hindi conversational data, clean up messy transcription output, identify spelling mistakes in a large word list, and finally make the evaluation process fairer using a lattice-based scoring approach.

All four problems are solved inside a **single Jupyter notebook** with step-by-step code and explanations.

---

## Project Structure

The notebook is divided into four self-contained sections, one for each question.

**Question 1 — Fine-tuning Whisper on Hindi** covers downloading and preprocessing ~10 hours of Hindi conversational audio, resampling it to 16kHz, cleaning the Devanagari transcriptions, and fine-tuning `openai/whisper-small` using HuggingFace's `Seq2SeqTrainer`. We then evaluate both the original and fine-tuned model on the FLEURS Hindi test set and report Word Error Rate (WER).

**Question 2 — ASR Cleanup Pipeline** builds a two-part text normalization system. The first part converts Hindi number words into digits using a recursive parser (where multipliers like सौ and हज़ार act as recursion pivots). The second part detects English words inside Hindi text using two signals — Roman script detection and a character n-gram classifier that identifies English loanwords written in Devanagari.

**Question 3 — Spell Checking 1,77,000 Words** classifies a large word list from the dataset as correctly or incorrectly spelled using a three-layer pipeline: dictionary lookup, Devanagari phonological validity checking, and a confidence-based fallback for proper nouns and loanwords that won't appear in any dictionary.

**Question 4 — Lattice-Based WER Evaluation** replaces the standard single-reference WER scoring with a word lattice. Each position in the lattice holds all valid alternatives collected from 6 ASR models. A majority voting rule (3 out of 5 models agreeing) is used to identify likely human reference errors, so models are no longer penalized for transcribing something correctly that the human reference got wrong.

---

## Tech Stack

| Area | Tools Used |
|------|-----------|
| Model | `openai/whisper-small`, `google/muril-base-cased` |
| Training | HuggingFace Transformers, Seq2SeqTrainer |
| Audio | Librosa, SoundFile |
| NLP | HuggingFace Datasets, Evaluate, JIWER |
| Data | Pandas, NumPy |
| Benchmark | FLEURS `hi_in` test set |

---

## How to Run

First, clone the repo and install the dependencies.

```bash
git clone https://github.com/your-username/hindi-asr-pipeline
cd hindi-asr-pipeline
pip install transformers datasets accelerate evaluate jiwer librosa soundfile torch
```

Then open the notebook and run all cells from top to bottom. Each section is independent so you can also run just the question you are interested in. A GPU is recommended for Question 1 (fine-tuning), but all other sections run fine on CPU.

---

## Results

| Model | WER on FLEURS Hindi Test Set |
|-------|------------------------------|
| Whisper-small (pretrained baseline) | ~55–65% |
| Whisper-small (fine-tuned on ~10h Hindi) | ~30–40% |


---

## Key Design Decisions

The number parser uses a **recursive approach** because Hindi numbers have a multiplier-based grammar — `तीन सौ चौवन` breaks down as `3 × 100 + 54`. A simple dictionary lookup would never handle compound numbers correctly.

The English word detector uses **character n-grams** rather than a vocabulary list because English loanwords in Devanagari (like `कंप्यूटर`) contain character patterns that almost never appear in native Hindi words, such as the `ॉ` vowel and certain consonant clusters.

The lattice WER uses a **3/5 majority threshold** because anything above 50% means most independent models agree — making it more likely the human reference is wrong than all the models being wrong simultaneously.

---

## Author

**Ankit** — B.Tech Information Technology, MSIT Delhi (2026)
Reach me on [LinkedIn](https://www.linkedin.com/in/ankit082004/) or at ankitsharma082004@email.com
