# 🛒 Walmart Demand Forecasting Dashboard

> End-to-end retail demand forecasting using LightGBM trained on 
> the Walmart M5 Kaggle dataset — deployed as an interactive dashboard.

🚀 **[Live Demo →](https://walmart-smart-demand-forecasting.streamlit.app/)**


---

## Results

| Submission | Private Score | Public Score |
|------------|--------------|--------------|
| Best submission | 0.74967 | **0.63772** |
| Initial submission | 0.77101 | 0.64573 |

*Evaluated on WRMSSE — lower is better.*
*Best public score of 0.638 represents a 47% improvement over naive baseline (~1.20).*

---

## What It Does

Forecasts daily sales for 3,049 Walmart items across 10 stores 
and 3 categories, 28 days ahead. The dashboard lets you:

- Select any store (CA/TX/WI), category, and item
- View 90 days of historical sales + 28-day forecast
- See real calendar event annotations (Easter, Mother's Day etc.)
- Get dynamic inventory restock recommendations

---

## Technical Highlights

**Model:** LightGBM with Tweedie objective  
Tweedie chosen because 60%+ of M5 items have zero-sale days — 
handles sparse count data better than MSE.

**Features (17 total):**
- Lag features at 7, 28, 35, 42, 49, 56 days
- Rolling mean/std over 7/14/28/60 day windows
- Price signals: momentum, week-over-week change, dept-relative price
- Calendar event flags

**Validation:** 3-fold walk-forward CV — train on past, 
validate on future. No data leakage.

**Scale:** 30M+ rows, memory optimized via dtype 
downcasting (70% RAM reduction).

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```


---

## Stack

Python · LightGBM · Pandas · NumPy · Plotly · Streamlit

---

*Built by Riya Setia · Data Science @ GD Goenka University*
