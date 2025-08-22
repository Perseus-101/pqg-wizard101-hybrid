# Procedural Quest Generation (Wizard101)

*MSc Thesis, University of Edinburgh (2025)*\
Author: **Denil Binu**

This repository contains the artefacts and code for the dissertation **“Procedural Quest Generation using Large Language Models (LLMs)”**. It compares a **Baseline** pipeline (schema-only prompting) with an **Advanced** pipeline that adds **structured indexes**, **RAG over lore**, and **explicit planning**, producing schema-valid JSON quests for Wizard101.

---

## ✨ What’s here

- `main.py` — Implements both pipelines (Baseline & Advanced), loads indexes + lore, builds a Chroma vector store, and handles prompts.
- `evaluate.py` — Runs the controlled experiment: generates 5 scenarios with both systems, validates JSON against the schema, and saves outputs.
- `environment.yml` — Exact **Conda** environment (Python 3.10.18) used for the dissertation experiments.
- `knowledge_base/`
  - `wizard101_quest_schema.json` — **Canonical quest schema** (stored at the \*\*root of \*\*\`\`).
  - `wizard_city/indexes_json/` — Structured **indexes** (NPCs, locations, monsters).
  - `wizard_city/lore_md/locations|npcs|story_arcs/` — Markdown **lore** used for RAG.
- `human_survey_outputs/`
  - `advanced_system/` — 5 generated quests (one per scenario).
  - `baseline_system/` — 5 generated quests (one per scenario).

---

## 🧪 Reproducibility snapshot

Environment (`environment.yml`):

- Name: `pqg_env`
- Python: **3.10.18**
- Key libs:\
  `openai==1.84.0`, `langchain==0.3.25`, `langchain-openai==0.3.21`,\
  `chromadb==1.0.12`, `jsonschema==4.24.0`, `tiktoken==0.9.0`,\
  `numpy==2.2.6`, `scikit-learn==1.7.0`, `certifi==2025.4.26`.

> The code sets `SSL_CERT_FILE` to `certifi.where()` to avoid SSL issues when embedding/retrieving lore.

---

## ⚙️ Setup

### 1) Create and activate the Conda env

```bash
conda env create -f environment.yml
conda activate pqg_env
```

### 2) Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY="..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="..."
```

### 3) Keep the expected layout

Run from the repo root with this relative structure intact:

```
knowledge_base/
  wizard101_quest_schema.json
  wizard_city/
    indexes_json/            # wc_monsters.json, wc_npcs.json, wc_locations.json
    lore_md/
      locations/
      npcs/
      story_arcs/
```

Outputs will go to:

```
human_survey_outputs/
  advanced_system/
  baseline_system/
```

---

## ▶️ How to run

Generate all 5 scenarios with **both** systems and validate against the schema:

```bash
python evaluate.py
```

You’ll see per-scenario logs and a validity flag. Results are saved under `human_survey_outputs/{advanced_system|baseline_system}/prompt_0X.json`.

---

## 🔧 Useful toggles (in `evaluate.py`)

Open `evaluate.py` and adjust:

- Which systems to run (both default to GPT-4o):

```python
SYSTEMS_TO_TEST = {
    "advanced_system": "gpt-4o",
    "baseline_system": "gpt-4o",
}
```

- How many scenarios to generate:

```python
NUM_QUESTS_TO_GENERATE = 5  # set to 1..5
```

- Scenario definitions (ids, level, location, last\_quest\_summary, trigger):

```python
PLAYER_SCENARIOS = [ {...}, ... ]  # five Wizard City scenarios
```

---

## 🧩 What the two systems do

- **Baseline**

  - Same model as Advanced (GPT-4o)
  - Prompt includes **only** the **full JSON schema** + the **natural-language request**
  - No indexes, no RAG, no explicit planning

- **Advanced**

  - Adds **structured indexes** (NPCs, locations, monsters) as static context
  - Builds a **Chroma** vector store over lore markdown and **retrieves** relevant passages (RAG)
  - Encourages **intermediate planning** before emitting final JSON
  - Same schema used for strict structural guidance

Both outputs are validated with `jsonschema` against `knowledge_base/wizard101_quest_schema.json`.

---

## 🧰 Troubleshooting

- **“FATAL: A required context file was not found”**\
  Ensure the schema and indexes are at the paths shown above.

- **Chroma store / write permissions**\
  The vector store writes a local folder. Run from a writable directory.

- **SSL or cert errors**\
  `main.py` sets `SSL_CERT_FILE` to `certifi` automatically. Update `certifi` if needed.

- **Rate limits / costs**\
  This code calls OpenAI APIs multiple times. Ensure your key has quota and billing.

---

## 📝 License & Citation

Released under the **MIT License** (see `LICENSE`).\
If you use this code or data, please cite the dissertation (see `CITATION.cff`).

