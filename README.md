# 🧠 LLM-CRYPTO-BENCH

Benchmarking framework for evaluating Large Language Models (LLMs) on cryptography-related reasoning and problem-solving tasks.

---

## ⚙️ Project Overview

This repository provides a unified interface for evaluating multiple LLMs (Gemini, LLaMA, Mixtral, etc.) across three standardized cryptography datasets:

- **CipherBank** – structured CTF-style tasks from Hugging Face.
- **CipherBench** – question/answer–based cipher tasks.
- **CyberMetric** – curated real-world multiple-choice questions.

Each dataset is processed through adapter modules, evaluated via model backends, and scored using uniform metrics.

---

## 🧩 Requirements

**Python Version (Important):**
> ✅ Use **Python 3.11.10** only.

All dependencies are verified with Python 3.11.10 to prevent incompatibilities with the new `google-genai` SDK and `pandas` release.

Check your version:

```bash
python3 --version
