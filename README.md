# Player Churn Prediction System

An end-to-end ML web app that predicts player churn risk for online gaming platforms — powered by Logistic Regression and Decision Tree models trained on 40 000+ records, with an AI-powered engagement optimizer built on top.

> **Live Demo:** _add Streamlit Cloud link here_

---

## Features

- Upload player CSV → get churn probability, risk tier (Low/Medium/High), and downloadable results
- Two models: Logistic Regression and Decision Tree
- AI Agent tab: personalized engagement recommendations via Claude Haiku
- PDF + JSON report export

---

## Quick Start

```bash
git clone https://github.com/<your-username>/Churn_Prediction.git
cd Churn_Prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

For the AI Agent, set your Anthropic API key first:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Project Structure

```
├── app.py                  # Streamlit entry point
├── src/
│   ├── preprocessing.py    # Feature encoding & imputation
│   ├── inference.py        # Model loading & prediction
│   ├── train.py            # Training pipeline (CV + metrics)
│   └── ui.py               # Streamlit UI components
├── agent/
│   ├── pipeline.py         # 6-stage agentic pipeline
│   ├── knowledge_base.py   # Retention strategy retrieval
│   ├── llm.py              # Claude Haiku integration
│   └── export.py           # PDF report generation
├── pages/
│   └── 2_Agent_Optimizer.py # Agent Streamlit page
├── tests/                  # 44 unit tests
├── models/                 # Trained model artefacts
├── data/                   # Dataset (download from Kaggle)
└── reports/report.tex      # LaTeX project report
```

---

## Retrain Models

```bash
python -m src.train
```

Saves updated models and `models/evaluation_metrics.json` with CV scores and feature importances.

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Input Schema

CSV must include these 11 columns (missing values auto-imputed):

| Column | Type | Values |
|--------|------|--------|
| `Age` | int | 18–60 |
| `Gender` | str | Male / Female |
| `Location` | str | USA / Europe / Asia / Other |
| `GameGenre` | str | Action / RPG / Simulation / Sports / Strategy |
| `PlayTimeHours` | float | ≥ 0 |
| `InGamePurchases` | int | 0 or 1 |
| `GameDifficulty` | str | Easy / Medium / Hard |
| `SessionsPerWeek` | int | ≥ 0 |
| `AvgSessionDurationMinutes` | float | ≥ 0 |
| `PlayerLevel` | int | ≥ 1 |
| `AchievementsUnlocked` | int | ≥ 0 |

---

## Deployment

**Streamlit Cloud:** Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → select `app.py` → Deploy.

Add `ANTHROPIC_API_KEY` in the Streamlit Cloud secrets panel for the AI agent.
