# Decomposing-Fairness-Failures
This project conducts a ML fairness audit on an income survey dataset with a Random Forest Regressor to predict total income from three features (i.e., work reference, education level, and province). By evaluating predictive accuracy, the project aims to distinguish structural biases rooted in data and algorithmic biases introduced by the model.

---

## Project Description

In implementing responsible AI, developers have an obligation to hold models accountable in data analysis and findings potentially influenced by biases, circumstances, and fairness (or lack thereof). The analysis below uses a Canadian income survey dataset (Hirapara, 2025) to predict individuals' total incomes to determine whether inferences indicate bias in their results; consequently, this leads to a deeper question: 

**How might we determine structural inequality in data and/or a model so we might differentiate structural versus algorithmic biases and use appropriate mitigation techniques?**

This project is inspired by the Digital Minds Expert Forecasts in 2025 detailing how increased capacity for welfare is an anticipated capability of AI systems; therefore, we must be able to determine where biases stem from to mitigate them, especially as potential subjectivity from these systems requires preparation (Caviola & Saad, 2025).

Using a Random Forest Regressor, the analysis below attempts to predict total income from three features: (1) the province of residence, (2), the highest education level earned, and (3) the work reference (i.e., employed, unemployed, or retired). The dataset was preprocessed by selecting these columns explicitly, and an 80/20 train-test split was applied before training took place. After applying a baseline evaluation with MAE and R², I chose to retrain the model using inverse-frequency sample weights by province and compare baseline and mitigated disparity scores across all three features; consequently, the model is told to pay more attention to underrepresented groups during training. Since the resulting analysis did not change the model's outputs, the model is showing that it is considering underrepresented groups but requires more information. As these features were selected deliberately to avoid sensitive data, the model is preserving fairness where possible, but this does not imply the model's results are fair.

Looking at the final results, the means of actual and predicted incomes are close, as they are within `$500` (`$111,127` versus `$111,443`). This is, however, misleading as the MAE is `$52,880` and median error is 40%. The model is inaccurate at the individual level, and it predits the average by chance by overcorrecting in both directions (as seen in the Shap analysis and final findings). For the low income earners in the dataset, the model predicts approximately `$94,000` for total income whereas the dataset has an average of `$35,000`; this is a 161% error and indicates overprediction. For the high income earners, the model's results show an average of `$125,000` while the dataset shows `$218,000`; this reveals underprediction. The distributional unfairness would not showcase the edgecases of income earners accurately, despite being reliable for middle income earners. 

What does this mean? In returning to the key question of differentiating between structural versus algorithmic fairness, there are a few key considerations for models like this one. Firstly, mitigation does not affect the data when the information provided to a model is not sufficient to properly make analysis-based decisions. This is structural versus algorithmic, which means more data is needed when sensitive data is permissible to use and responsibly used for analysis. Secondly, the low income earner overprediction is not due to the model but the same need for more information for edgecases, so the model would benefit from having more data on groups which show more variability to determine whether that is truly accurate. Lastly, the features selected for this analysis are important, because they consider how, from an administrative perspective, income earners would be perceived. In reality, people experience income in different ways, so an analysis like this one should not be used in isolation based on this third indication of structural bias.

Overall, this exploration reveals that establishing baseline disparity can reveal structural versus algorithmic biases to make sure that, for all datasets and edgecases within, populations are adequately represented in data and by models.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/income-fairness-audit.git
cd income-fairness-audit
pip install -r requirements.txt
```

Required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

> SHAP is optional. The explainability section (Section 7) will skip gracefully if it is not installed.

---

## Data

**Source:** [Income Survey Finance Analysis — Kaggle](https://www.kaggle.com/datasets/aradhanahirapara/income-survey-finance-analysis)

> This dataset is a Canadian public income survey containing financial and demographic information at the individual level.

---

## Limitations

- **Narrow Feature Set:** Only four features were used, and these were deliberately selected to avoid sensitive demographic variables. This limits predictive power, particularly for edgecases.
- **Historical Data:** The dataset reflects historical income patterns.
- **Responsible AI Tradeoff:** The decision to exclude sensitive demographic variables was made deliberately. This represents a real fairness-preserving feature selection cost and a tradeoff that is itself a finding of the project.

---

## Bibliography

Caviola, L., & Saad, B. (2025). Expert Forecasts in 2025 | Futures with Digital Minds. Expert Forecasts in 2025. https://digitalminds.report/forecasting-2025/#social-function 

Hirapara, A. (2025, March 20). Income survey: Finance analysis. Kaggle. https://www.kaggle.com/datasets/aradhanahirapara/income-survey-finance-analysis 

---
