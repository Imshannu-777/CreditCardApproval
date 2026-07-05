import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD CLEANED DATASET
# ============================================================
df = pd.read_csv('cleaned_data.csv')
print("Dataset Shape:", df.shape)
print("\nClass Balance (original):")
print(df['STATUS_BIN'].value_counts(normalize=True))

X = df.drop(columns=['ID', 'STATUS_BIN'])
y = df['STATUS_BIN']

# ============================================================
# STEP 2: STRATIFIED TRAIN/TEST SPLIT (before SMOTE!)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining Set Size (before SMOTE):", X_train.shape)
print("Testing Set Size:", X_test.shape)

# ============================================================
# STEP 3: APPLY SMOTE TO TRAINING DATA ONLY
# ============================================================
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Training Set Size (after SMOTE):", X_train_res.shape)
print("Train class balance after SMOTE:")
print(pd.Series(y_train_res).value_counts(normalize=True))

# ============================================================
# STEP 4: MODEL TRAINING & EVALUATION FUNCTION
# ============================================================
results = {}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    print("\nClassification Report (on REAL, untouched test set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

    results[name] = f1
    return model, f1

# Note: We train on SMOTE-resampled data, but ALWAYS evaluate on the
# real, untouched, original-distribution test set (X_test, y_test).

# ============================================================
# STEP 5: LOGISTIC REGRESSION
# ============================================================
lr_model, lr_f1 = evaluate_model(
    "Logistic Regression",
    LogisticRegression(random_state=42, max_iter=1000),
    X_train_res, X_test, y_train_res, y_test
)

# ============================================================
# STEP 6: DECISION TREE
# ============================================================
dt_model, dt_f1 = evaluate_model(
    "Decision Tree",
    DecisionTreeClassifier(random_state=42, max_depth=10),
    X_train_res, X_test, y_train_res, y_test
)

# ============================================================
# STEP 7: RANDOM FOREST
# ============================================================
rf_model, rf_f1 = evaluate_model(
    "Random Forest",
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=12),
    X_train_res, X_test, y_train_res, y_test
)

# ============================================================
# STEP 8: XGBOOST (Gradient Boosting)
# ============================================================
xgb_model, xgb_f1 = evaluate_model(
    "XGBoost",
    GradientBoostingClassifier(random_state=42),
    X_train_res, X_test, y_train_res, y_test
)

# ============================================================
# STEP 9: MODEL COMPARISON
# ============================================================
print("\n" + "="*50)
print("   MODEL COMPARISON SUMMARY (Macro F1-score)")
print("="*50)
for model_name, f1 in results.items():
    print(f"{model_name:<25} F1: {f1:.4f}")

best_model_name = max(results, key=results.get)
best_f1 = results[best_model_name]
print(f"\n✅ Best Model: {best_model_name} with Macro F1: {best_f1:.4f}")

# ============================================================
# STEP 10: SAVE BEST MODEL
# ============================================================
best_models = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

best_model = best_models[best_model_name]
joblib.dump(best_model, 'model.pkl')
print(f"\n✅ Best model saved as model.pkl")

feature_cols = list(X.columns)
joblib.dump(feature_cols, 'feature_columns.pkl')
print("✅ Feature columns saved as feature_columns.pkl")

# ============================================================
# STEP 11: COMPARISON CHART
# ============================================================
plt.figure(figsize=(10,6))
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
bars = plt.bar(results.keys(), results.values(), color=colors)
plt.title('Model Macro F1-score Comparison (SMOTE-trained)', fontsize=16)
plt.ylabel('Macro F1-score', fontsize=12)
plt.ylim(0, 1.0)
for bar, f1 in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{f1:.4f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("✅ Model comparison chart saved as model_comparison.png")