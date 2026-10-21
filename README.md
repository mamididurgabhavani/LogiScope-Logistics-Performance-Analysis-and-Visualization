# LogiScope: Logistics Analysis and Optimization

Python project on a simulated logistics dataset (1,500 shipments, 5 hubs, 4 carriers, 4 transport modes).

## Week 3: Data analysis and visualization
- `logistics_analysis.py` simulates the data, runs exploratory analysis and produces eight charts.

## Week 4: Predictive modeling and optimization
- `logistics_model.py` predicts shipment delay (Linear/Ridge Regression, Decision Tree, Random Forest, Gradient Boosting), validates with 5-fold cross-validation and grid search, then:
  - allocates shipments to carriers with linear programming (SciPy HiGHS), and
  - chooses a risk threshold for proactive intervention.
- Test-set result for the tuned Gradient Boosting model: RMSE 0.63 days, MAE 0.47 days, R-squared 0.47.

## Files
| File | Purpose |
|---|---|
| `logistics_analysis.py` | Week 3 simulation, EDA and charts (writes `logistics_data.csv`) |
| `logistics_model.py` | Week 4 modeling and optimization (reads `logistics_data.csv`) |
| `logistics_data.csv` | Simulated shipment dataset |
| `requirements.txt` | Python dependencies |

## Run
```bash
pip install -r requirements.txt
mkdir charts charts4
python logistics_analysis.py
python logistics_model.py
```

Costs used in the optimization (delay penalty, intervention cost and effectiveness) are stated assumptions, not real company data.
