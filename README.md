# ClinIQ — Clinical Patient Risk Intelligence System

## Setup

```bash
pip install -r requirements.txt
```

## Dataset
Download `mtsamples.csv` from:  
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions  
Place it in the project root folder.

## Run the notebook
```bash
jupyter notebook cliniq_notebook.ipynb
```
Run all cells top-to-bottom. This will:
- Train the XGBoost model
- Compute SHAP values
- Build the FAISS RAG index
- Run the Fairlearn bias audit
- Save `cliniq_artefacts.pkl` and all plots to `results/`

## Run the Streamlit dashboard
```bash
streamlit run app.py
```
Make sure `cliniq_artefacts.pkl` is in the same folder.

## Files
| File | Description |
|------|-------------|
| `cliniq_notebook.ipynb` | Full end-to-end pipeline notebook |
| `app.py` | 4-tab Streamlit dashboard |
| `cliniq_artefacts.pkl` | Pre-trained model + all artefacts |
| `requirements.txt` | Python dependencies |

## Tabs in the Dashboard
1. **Risk Predictor** — paste any clinical note, get risk level + SHAP explanation
2. **RAG Clinical Q&A** — semantic retrieval + RoBERTa answer extraction
3. **Model Performance** — confusion matrix, confidence plots, feature importances
4. **Bias Audit** — Fairlearn MetricFrame across gender and age groups
