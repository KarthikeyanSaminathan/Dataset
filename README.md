# Formula 1 Circuits - Simple ML Prediction Model

A simple and quick Machine Learning model built on `circuits.csv` that predicts a circuit's **Geographic Region** (Europe, Americas, Asia-Oceania, Africa) from its GPS coordinates (`lat`, `lng`).

## How to Run

Open your terminal in this folder and run:

```bash
python simple_model.py
```

## Model Outcome
- **Algorithm**: Random Forest Classifier
- **Features**: Latitude (`lat`), Longitude (`lng`)
- **Target**: Region (`Europe`, `Americas`, `Asia-Oceania`, `Africa`)
- **Accuracy**: **93.33%**
- **Output Plot**: Saved automatically as `simple_model_outcome.png`

