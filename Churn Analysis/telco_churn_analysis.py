import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("TELCO CUSTOMER CHURN PREDICTION ANALYSIS")
print("="*60)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"   Data shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Initial data exploration
print("\n2. Initial Data Exploration...")
print(f"   Missing values:\n{df.isnull().sum()}")
print(f"\n   Data types:\n{df.dtypes}")
print(f"\n   First few rows:")
print(df.head())

# Check for any issues with TotalCharges
print("\n3. Checking TotalCharges column...")
print(f"   TotalCharges unique values (first 20): {df['TotalCharges'].unique()[:20]}")
# Check if TotalCharges has any non-numeric values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"   Missing values in TotalCharges after conversion: {df['TotalCharges'].isnull().sum()}")

# Data Cleaning
print("\n4. Data Cleaning...")
# Drop customerID as it's not useful for prediction
df_clean = df.drop('customerID', axis=1)

# Handle missing values in TotalCharges (replace with 0 or median)
if df_clean['TotalCharges'].isnull().sum() > 0:
    df_clean['TotalCharges'].fillna(0, inplace=True)
    print(f"   Filled {df['TotalCharges'].isnull().sum()} missing TotalCharges values with 0")

# Encode target variable
le_target = LabelEncoder()
df_clean['Churn'] = le_target.fit_transform(df_clean['Churn'])
print(f"   Encoded Churn: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Separate features and target
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\n   Categorical columns: {categorical_cols}")
print(f"   Numerical columns: {numerical_cols}")

# Encode categorical variables
print("\n5. Encoding categorical variables...")
X_encoded = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le
    print(f"   Encoded {col}: {len(le.classes_)} unique values")

# Check class distribution
print("\n6. Class Distribution:")
print(f"   Churn = No (0): {(y == 0).sum()} ({(y == 0).sum()/len(y)*100:.2f}%)")
print(f"   Churn = Yes (1): {(y == 1).sum()} ({(y == 1).sum()/len(y)*100:.2f}%)")

# Split the data
print("\n7. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Scale numerical features
print("\n8. Scaling numerical features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ============================================
# LOGISTIC REGRESSION MODEL
# ============================================
print("\n" + "="*60)
print("LOGISTIC REGRESSION MODEL")
print("="*60)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_precision = precision_score(y_test, y_pred_lr)
lr_recall = recall_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_roc_auc = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\nLogistic Regression Metrics:")
print(f"  Accuracy:  {lr_accuracy:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall:    {lr_recall:.4f}")
print(f"  F1-Score:  {lr_f1:.4f}")
print(f"  ROC-AUC:   {lr_roc_auc:.4f}")

# Feature importance for Logistic Regression
lr_coef = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_[0]
})
lr_coef['Abs_Coefficient'] = np.abs(lr_coef['Coefficient'])
lr_coef = lr_coef.sort_values('Abs_Coefficient', ascending=False)

print(f"\nTop 10 Most Important Features (Logistic Regression):")
print(lr_coef.head(10)[['Feature', 'Coefficient']].to_string(index=False))

# ============================================
# RANDOM FOREST MODEL
# ============================================
print("\n" + "="*60)
print("RANDOM FOREST MODEL")
print("="*60)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_roc_auc = roc_auc_score(y_test, y_pred_proba_rf)

print(f"\nRandom Forest Metrics:")
print(f"  Accuracy:  {rf_accuracy:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"  ROC-AUC:   {rf_roc_auc:.4f}")

# Feature importance for Random Forest
rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})
rf_importance = rf_importance.sort_values('Importance', ascending=False)

print(f"\nTop 10 Most Important Features (Random Forest):")
print(rf_importance.head(10).to_string(index=False))

# ============================================
# MODEL COMPARISON
# ============================================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc],
    'Random Forest': [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc]
})

comparison['Difference'] = comparison['Random Forest'] - comparison['Logistic Regression']
comparison['Winner'] = comparison.apply(
    lambda x: 'Random Forest' if x['Difference'] > 0 else 'Logistic Regression', axis=1
)

print("\nDetailed Comparison:")
print(comparison.to_string(index=False))

# Determine overall winner
scores_lr = [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc]
scores_rf = [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc]

lr_wins = sum(1 for i in range(len(scores_lr)) if scores_lr[i] > scores_rf[i])
rf_wins = sum(1 for i in range(len(scores_lr)) if scores_rf[i] > scores_lr[i])

print(f"\n{'='*60}")
print("OVERALL WINNER:")
if rf_wins > lr_wins:
    print("  *** RANDOM FOREST is the better model! ***")
    print(f"  Random Forest wins in {rf_wins} out of 5 metrics")
elif lr_wins > rf_wins:
    print("  *** LOGISTIC REGRESSION is the better model! ***")
    print(f"  Logistic Regression wins in {lr_wins} out of 5 metrics")
else:
    print("  *** It's a tie! Both models perform similarly. ***")
print(f"{'='*60}")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n9. Creating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.35
ax1.bar(x - width/2, [lr_accuracy, lr_precision, lr_recall, lr_f1, lr_roc_auc], 
        width, label='Logistic Regression', alpha=0.8)
ax1.bar(x + width/2, [rf_accuracy, rf_precision, rf_recall, rf_f1, rf_roc_auc], 
        width, label='Random Forest', alpha=0.8)
ax1.set_xlabel('Metrics')
ax1.set_ylabel('Score')
ax1.set_title('Model Comparison: All Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. ROC Curves
ax2 = plt.subplot(2, 3, 2)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
ax2.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_roc_auc:.4f})', linewidth=2)
ax2.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_roc_auc:.4f})', linewidth=2)
ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves Comparison')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Confusion Matrix - Logistic Regression
ax3 = plt.subplot(2, 3, 3)
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
ax3.set_title(f'Confusion Matrix - Logistic Regression\nAccuracy: {lr_accuracy:.4f}')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')

# 4. Confusion Matrix - Random Forest
ax4 = plt.subplot(2, 3, 4)
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax4,
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
ax4.set_title(f'Confusion Matrix - Random Forest\nAccuracy: {rf_accuracy:.4f}')
ax4.set_ylabel('Actual')
ax4.set_xlabel('Predicted')

# 5. Feature Importance - Logistic Regression
ax5 = plt.subplot(2, 3, 5)
top_features_lr = lr_coef.head(10)
ax5.barh(range(len(top_features_lr)), top_features_lr['Abs_Coefficient'])
ax5.set_yticks(range(len(top_features_lr)))
ax5.set_yticklabels(top_features_lr['Feature'])
ax5.set_xlabel('Absolute Coefficient Value')
ax5.set_title('Top 10 Features - Logistic Regression')
ax5.invert_yaxis()
ax5.grid(axis='x', alpha=0.3)

# 6. Feature Importance - Random Forest
ax6 = plt.subplot(2, 3, 6)
top_features_rf = rf_importance.head(10)
ax6.barh(range(len(top_features_rf)), top_features_rf['Importance'], color='green')
ax6.set_yticks(range(len(top_features_rf)))
ax6.set_yticklabels(top_features_rf['Feature'])
ax6.set_xlabel('Importance Score')
ax6.set_title('Top 10 Features - Random Forest')
ax6.invert_yaxis()
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('telco_churn_model_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved visualization: telco_churn_model_comparison.png")

# Additional detailed comparison plot
fig2, ax = plt.subplots(figsize=(10, 6))
comparison_melted = comparison.melt(id_vars='Metric', 
                                    value_vars=['Logistic Regression', 'Random Forest'],
                                    var_name='Model', value_name='Score')
sns.barplot(data=comparison_melted, x='Metric', y='Score', hue='Model', ax=ax)
ax.set_title('Detailed Model Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Metric', fontsize=12)
ax.legend(title='Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('telco_churn_detailed_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved visualization: telco_churn_detailed_comparison.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nSummary:")
print(f"  • Logistic Regression ROC-AUC: {lr_roc_auc:.4f}")
print(f"  • Random Forest ROC-AUC: {rf_roc_auc:.4f}")
print(f"  • Best Model: {'Random Forest' if rf_roc_auc > lr_roc_auc else 'Logistic Regression'}")
print(f"\nVisualizations saved in the current directory.")

