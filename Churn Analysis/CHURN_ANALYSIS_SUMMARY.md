# Telco Customer Churn Prediction Analysis - Summary Report

## Executive Summary

This analysis compares **Logistic Regression** and **Random Forest** models for predicting customer churn in the Telco dataset. After comprehensive evaluation, **Logistic Regression emerges as the better model**, winning in 4 out of 5 key metrics.

---

## Dataset Overview

- **Total Records**: 7,043 customers
- **Features**: 20 predictive features + 1 target variable (Churn)
- **Class Distribution**:
  - No Churn: 5,174 (73.46%)
  - Churn: 1,869 (26.54%)

### Data Cleaning Performed

1. **Removed** `customerID` (identifier, not predictive)
2. **Converted** `TotalCharges` from string to numeric
3. **Handled** 11 missing values in `TotalCharges` (filled with 0)
4. **Encoded** all categorical variables using Label Encoding
5. **Scaled** numerical features using StandardScaler
6. **Split** data: 80% training (5,634 samples), 20% testing (1,409 samples)

---

## Model Performance Comparison

| Metric | Logistic Regression | Random Forest | Winner |
|--------|-------------------|---------------|--------|
| **Accuracy** | 0.7984 | 0.7956 | **Logistic Regression** |
| **Precision** | 0.6406 | 0.6453 | **Random Forest** |
| **Recall** | 0.5481 | 0.5107 | **Logistic Regression** |
| **F1-Score** | 0.5908 | 0.5701 | **Logistic Regression** |
| **ROC-AUC** | 0.8404 | 0.8350 | **Logistic Regression** |

### Overall Winner: **Logistic Regression** (4 out of 5 metrics)

---

## Key Insights

### Most Important Features (Both Models Agree)

1. **Tenure** - Customer's length of service (most important)
2. **MonthlyCharges** - Monthly subscription cost
3. **Contract** - Contract type (Month-to-month, One year, Two year)
4. **TotalCharges** - Total amount charged to customer
5. **OnlineSecurity** - Whether customer has online security
6. **TechSupport** - Whether customer has tech support

### Model-Specific Insights

#### Logistic Regression
- **Strengths**: Better at identifying customers who will churn (higher recall)
- **Best Use Case**: When you want to minimize false negatives (missed churners)
- **Top Feature**: Tenure has the strongest negative coefficient (-1.23), meaning longer tenure = lower churn risk

#### Random Forest
- **Strengths**: Slightly better precision (fewer false positives)
- **Best Use Case**: When you want to minimize false positives (incorrectly flagged as churners)
- **Top Feature**: Also identifies tenure as most important, but with more balanced feature importance distribution

---

## Recommendations

### For Business Use

1. **Use Logistic Regression** as the primary model due to its superior overall performance
2. **Focus on tenure** - This is the strongest predictor across both models
3. **Monitor contract types** - Month-to-month contracts are higher risk
4. **Review pricing strategy** - Higher monthly charges correlate with churn
5. **Enhance service offerings** - Online security and tech support reduce churn

### Model Deployment Strategy

- **Primary Model**: Logistic Regression (better recall = catch more churners)
- **Secondary Check**: Use Random Forest for precision validation if needed
- **Threshold Tuning**: Consider adjusting classification threshold based on business cost of false positives vs. false negatives

---

## Files Generated

1. **telco_churn_analysis.py** - Complete analysis script
2. **telco_churn_model_comparison.png** - Comprehensive visualization dashboard
3. **telco_churn_detailed_comparison.png** - Detailed metric comparison chart
4. **CHURN_ANALYSIS_SUMMARY.md** - This summary report

---

## Technical Details

### Model Configurations

- **Logistic Regression**: 
  - Max iterations: 1000
  - Regularization: Default L2
  - Features scaled using StandardScaler

- **Random Forest**:
  - Number of trees: 100
  - Max depth: 10
  - Random state: 42 (for reproducibility)

### Evaluation Methodology

- **Train/Test Split**: 80/20 with stratification
- **Metrics Used**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-validation**: Not performed (can be added for more robust evaluation)

---

## Next Steps (Optional Enhancements)

1. **Hyperparameter Tuning**: Use GridSearchCV to optimize both models
2. **Feature Engineering**: Create interaction terms or polynomial features
3. **Class Imbalance Handling**: Try SMOTE or other techniques (though current performance is good)
4. **Cross-Validation**: Add k-fold cross-validation for more robust evaluation
5. **Ensemble Methods**: Combine both models for potentially better performance
6. **Business Metrics**: Incorporate cost-benefit analysis for false positives/negatives

---

*Analysis completed on: March 10, 2026*
*Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv*

