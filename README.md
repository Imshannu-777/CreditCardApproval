# 💳 Credit Card Approval Prediction

An end-to-end Machine Learning web application that predicts whether a credit card application will be **approved or rejected** based on applicant financial and demographic data — built with Python, Flask, and Scikit-learn.

---

## 🚀 Live Demo
👉 [Click here to try the app](https://tactile-dejected-lankiness.ngrok-free.dev)  if it is not work
> Clone the repo and run locally using Flask (see setup instructions below)

---

## 📌 Project Overview

Banks receive thousands of credit card applications daily. Manual review is time-consuming and error-prone. This system automates the approval decision using machine learning by evaluating applicant profiles — just like real banks do.

Four classification algorithms are trained and compared:
- Logistic Regression
- Decision Tree
- Random Forest *(best performer — 71% accuracy)*
- XGBoost (Gradient Boosting)

The best model is integrated into a Flask web application for real-time predictions through an intuitive UI.

---

## 🎯 Use Cases

| Scenario | Description |
|---|---|
| **Automated Screening** | Bank analyst enters applicant profile → instant approval/rejection prediction |
| **Risk Identification** | Compliance officer screens high-risk applicants with overdue records |
| **Self-Service Check** | Prospective customer checks eligibility before formally applying |

---

## 🗂️ Project Structure

```
CreditCardApproval/
│
├── templates/
│   ├── home.html          # Landing page
│   ├── index.html         # Application form
│   └── result.html        # Prediction result with confidence gauge
│
├── static/                # Static assets
│
├── data_preprocessing.py  # Data cleaning, merging, feature engineering
├── model_training.py      # Model training, evaluation, SMOTE balancing
├── app.py                 # Flask web application
├── feature_columns.pkl    # Saved feature column order
├── model_comparison.png   # Model accuracy comparison chart
└── confusion_matrix_*.png # Confusion matrices for all 4 models
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| ML Libraries | Scikit-learn, XGBoost, Imbalanced-learn (SMOTE) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Model Storage | Joblib |
| IDE | VS Code |

---

## 📊 Dataset

**Source:** [Credit Card Approval Prediction — Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

Two CSV files:
- `application_record.csv` — applicant demographic and financial details
- `credit_record.csv` — monthly credit payment status records

> ⚠️ Dataset files are not included in this repo due to size. Download from the Kaggle link above and place both CSVs in the project root before running.

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Imshannu-777/CreditCardApproval.git
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn flask joblib xgboost
```

### 3. Download dataset
Download from [Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) and place `application_record.csv` and `credit_record.csv` in the project folder.

### 4. Run preprocessing
```bash
python data_preprocessing.py
```

### 5. Train models
```bash
python model_training.py
```

### 6. Launch the web app
```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 🔄 ML Pipeline

```
Raw Data → Data Cleaning → Feature Engineering → Label Encoding
    → SMOTE Balancing → Train/Test Split → Model Training
        → Model Comparison → Best Model Saved → Flask Deployment
```

---

## 📈 Model Performance

| Model | Macro F1-Score | Accuracy |
|---|---|---|
| Logistic Regression | 0.659 | 67.7% |
| Decision Tree | 0.658 | 67.1% |
| **Random Forest** | **0.693** | **71.0%** |
| XGBoost | 0.669 | 69.1% |

> **Random Forest** selected as best model based on Macro F1-score.
> SMOTE oversampling applied to handle class imbalance during training.

---

## 🖥️ Application Screenshots

### Home Page
Clean fintech-style landing page with project overview and key stats.

### Application Form
Sectioned form collecting Personal Details, Financial Profile, Contact Information, and Credit Account History.

### Result Page
Animated circular confidence gauge showing approval probability with color-coded verdict (green = Approved, red = Rejected).

---

## 🧠 Key Concepts Demonstrated

-  Exploratory Data Analysis (Univariate, Multivariate, Descriptive)
-  Data Cleaning & Merging two datasets
-  Feature Engineering from multi-month credit records
-  Handling class imbalance with SMOTE
-  Training & comparing 4 classification algorithms
-  Model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
-  Flask web application with Jinja2 templating
-  Real-time ML model inference via REST-style form submission

---

## 👥 Team

| Role | Member |
|---|---|
| Team Lead | Me |


---

## 📄 License

This project is for educational demonstration purposes only.

---

⭐ If you found this useful, please star the repository!
