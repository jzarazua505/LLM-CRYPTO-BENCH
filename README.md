# LLM-CRYPTO-BENCH

Benchmarking framework for evaluating Large Language Models (LLMs) on cryptography-related reasoning and problem-solving tasks.

## Project Overview

This repository provides a unified environment for evaluating multiple LLMs (Gemini, LLaMA, Mixtral, GPT-OSS-20B, etc.) across three standardized cryptography datasets:

CipherBench — Instruction-following cipher puzzles requiring strict one-line outputs.

CipherBank — Classical cipher decryption tasks (Caesar, Atbash, Vigenère, and others).

CyberMetric — Multiple-choice cybersecurity and privacy questions.

Each dataset is processed through adapter modules, evaluated through a shared modeling interface, and scored with consistent, transparent metrics.

## Repository Structure

```bash
LLM-CRYPTO-BENCH/
├── adapters/
│   ├── base.py
│   ├── cipherbank.py
│   ├── cipherbench.py
│   └── cybermetric.py
│
├── datasets/
│   ├── cipherbank/
│   │   └── cipherbank.jsonl
│   ├── cipherbench/
│   │   └── cipherbench.jsonl
│   └── cybermetric/
│       └── cybermetric.json
│
├── eval/
│   └── run_eval.py
│
├── models/
│   ├── base.py
│   ├── google_gemini.py
│   ├── openrouter_client.py
│   ├── openrouter_gptoss.py
│   ├── openrouter_llama.py
│   ├── openrouter_mixtral.py
│   └── rate_limit.py
│
├── prompts/
│
├── results/
│
├── run_eval.ipynb
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```
## Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/<username>/LLM-CRYPTO-BENCH.git
cd LLM-CRYPTO-BENCH
```
2. Set your remote with your username
```bash
git remote set-url origin https://<YOUR_GITHUB_USERNAME>@github.com/jzarazua505/LLM-CRYPTO-BENCH.git
```
3. Create a GitHub Personal Access Token (PAT)

GitHub passwords do NOT work for pushing.

- Go to: https://github.com/settings/personal-access-tokens
- Generate a Fine-grained token
- Give it access to the LLM-CRYPTO-BENCH repo
- Permissions:
  - Contents → Read and write
  - Metadata → Read-only
- Copy the token (you won’t see it again)
4. Install Python 3.11.10 (Required)

This project must be run using Python 3.11.10.

Using Python 3.12+ will cause errors with the Gemini SDK and other dependencies.

macOS / Linux (recommended)
```bash
brew install pyenv
pyenv install 3.11.10
pyenv local 3.11.10
```
Windows (recommended)
```bash
winget install pyenv-win
pyenv install 3.11.10
pyenv local 3.11.10
```
After this step, entering the project directory will automatically activate Python 3.11.10 (due to the .python-version file).

4. Create and activate the virtual environment
```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
```
6. Install project dependencies
```bash
pip install -r requirements.txt
```
7. Select the correct Python interpreter in VS Code

VS Code will not automatically use the new environment.
You must manually select the correct interpreter:

- Open the project in VS Code
- Press Cmd+Shift+P (macOS) or Ctrl+Shift+P
- Select: Python: Select Interpreter
- Choose:
```bash
Python 3.11.10 ('venv')
```

8. Select the correct Jupyter kernel (for the notebook)
```bash
Python 3.11.10 ('venv')
```

9. Creat a .env like this
```bash
# ===== Keys & Model Names =====
GEMINI_API_KEY=
GEMINI_MODEL_NAME="gemini-2.5-flash-lite"

GITHUB_TOKEN=

# ===== Evaluation Behavior =====
EVAL_CONTENT_RETRIES=2
EVAL_CONTENT_BACKOFF=2.0
EVAL_MIN_SECS_BETWEEN_CALLS=3

# ===== OpenRouter Settings =====
OPENROUTER_API_KEY=
OPENROUTER_SITE_URL=https://github.com/jzarazua505/LLM-CRYPTO-BENCH
OPENROUTER_APP_NAME=LLM-CRYPTO-BENCH

# ===== Global Rate Limits =====
# Default for any model not specifically overridden
OPENROUTER_MAX_RPM=10


OPENROUTER_RPM_MODEL_openai_gpt_oss_20b_free=10

# Stronger floors when provider sends headerless 429s
OPENROUTER_429_FLOOR_SECONDS=20
OPENROUTER_429_COOLDOWN_SECONDS=60
```










