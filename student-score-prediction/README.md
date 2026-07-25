# Student Score Prediction System

A machine learning system that predicts student examination scores based on study
habits and academic indicators. Built as part of the Codomax Digital Solutions
AI & ML Internship, extended into a full engineering project with model
experimentation, a served API, automated testing, CI/CD, and cloud deployment.

## Project Goal

Go beyond a single notebook: build a complete, production-style ML product —
from raw data to a live, deployed prediction service — following the same
practices used on real engineering teams.

**End state:** a trained model served via a REST API, consumed by a working
front-end, fully tested, containerized, and deployed on Microsoft Azure.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data & ML | pandas, NumPy, scikit-learn |
| Experiment Tracking | MLflow |
| Visualization | Matplotlib, Plotly/Streamlit |
| API | FastAPI, Pydantic |
| Front-end | Streamlit / React |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Deployment | Azure App Service / Azure Container Apps |
| Monitoring | Azure Application Insights |

## Project Structure

```
student-score-prediction/
├── src/            # Reusable Python modules (data loading, cleaning, models, API)
├── tests/          # pytest unit tests
├── notebooks/      # Exploratory data analysis and prototyping
├── config/         # Configuration files (paths, hyperparameters)
├── data/
│   ├── raw/        # Original, untouched dataset
│   └── processed/  # Cleaned, feature-engineered data
├── requirements.txt
└── README.md
```

## Roadmap

- [x] Day 1 — Environment setup, project structure, first commit
- [x] Day 2 — Python fundamentals as tested utilities (`src/utils.py`, `tests/test_utils.py`)
- [ ] Day 3 — NumPy, from-scratch linear regression
- [ ] Day 4–5 — Data loading, cleaning, feature engineering
- [ ] Day 6 — Exploratory data analysis + interactive dashboard
- [ ] Day 7–8 — Model training, comparison, and experiment tracking (MLflow)
- [ ] Day 9 — FastAPI prediction service
- [ ] Day 10 — Model evaluation, explainability (SHAP), model card
- [ ] Day 11 — Front-end app consuming the API
- [ ] Day 12 — Testing, CI pipeline, Dockerfile
- [ ] Day 13 — GitHub documentation polish
- [ ] Day 14 — Final submission
- [ ] Day 15 (bonus) — Live Azure deployment

## Setup

```bash
# Clone the repo and navigate to this project
cd student-score-prediction

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Author

Malaika Malik — AI & ML Intern, Codomax Digital Solutions