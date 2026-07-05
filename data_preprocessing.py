import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ============================================================
# STEP 1: LOAD DATASETS
# ============================================================
app = pd.read_csv('application_record.csv')
credit = pd.read_csv('credit_record.csv')

print("Application Record Shape:", app.shape)
print("Credit Record Shape:", credit.shape)
print("\nApplication Record Head:")
print(app.head())

# ============================================================
# STEP 2: UNIVARIATE ANALYSIS
# ============================================================
print("\nOccupation Type Value Counts:")
print(app['OCCUPATION_TYPE'].value_counts())

plt.figure(figsize=(18,6))
sns.countplot(x='OCCUPATION_TYPE', data=app, palette='Set2')
plt.xticks(rotation=45, ha='right')
plt.title('Occupation Type Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='NAME_INCOME_TYPE', data=app, palette='Set1')
plt.title('Income Type Distribution')
plt.tight_layout()
plt.show()

# ============================================================
# STEP 3: MULTIVARIATE ANALYSIS
# ============================================================
plt.figure(figsize=(12,8))
numeric_app = app.select_dtypes(include=[np.number])
sns.heatmap(numeric_app.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# ============================================================
# STEP 4: DESCRIPTIVE ANALYSIS
# ============================================================
print("\nDescriptive Statistics:")
print(app.describe())

# ============================================================
# STEP 5: DROP DUPLICATES
# ============================================================
app.drop_duplicates(subset=['ID'], keep='first', inplace=True)
print("\nAfter dropping duplicates:", app.shape)

# ============================================================
# STEP 6: HANDLE MISSING VALUES
# ============================================================
print("\nMissing Values:")
print(app.isnull().mean())

# Drop occupation type (has missing values, not needed)
app.drop(columns=['OCCUPATION_TYPE'], inplace=True)

# ============================================================
# STEP 7: DATA CLEANING & FEATURE ENGINEERING
# ============================================================
def clean_application(df):
    df = df.copy()

    # Convert negative days to positive
    df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].abs()

    # Create AGE column (in years)
    df['AGE'] = (df['DAYS_BIRTH'] / 365).astype(int)

    # Create YEARS_EMPLOYED column
    df['YEARS_EMPLOYED'] = (df['DAYS_EMPLOYED'] / 365).astype(int)

    # Family dependency feature
    df['FAMILY_MEMBERS'] = df['CNT_FAM_MEMBERS'].fillna(0)
    df['CHILDREN'] = df['CNT_CHILDREN'].fillna(0)

    # Drop unwanted columns
    df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED',
                     'CNT_FAM_MEMBERS', 'CNT_CHILDREN'], inplace=True)

    return df

app = clean_application(app)

# ============================================================
# STEP 8: PROCESS CREDIT RECORDS
# ============================================================
def to_binary(status):
    if status in ['X', 'C']:
        return 1   # No active loan or fully paid -> Approved
    else:
        return 0   # Any overdue days (0-5) -> Not Approved   # Overdue / bad debt -> Not Approved

credit['STATUS_BIN'] = credit['STATUS'].apply(to_binary)
print("\nCredit Status Binary Counts:")
print(credit['STATUS_BIN'].value_counts())

# Group by ID - get approval status per applicant
credit_grouped = credit.groupby('ID')['STATUS_BIN'].mean().reset_index()
credit_grouped['STATUS_BIN'] = (credit_grouped['STATUS_BIN'] >= 0.5).astype(int)

# ============================================================
# STEP 9: MERGE DATASETS
# ============================================================
final_df = app.merge(credit_grouped, on='ID', how='left')
final_df.dropna(subset=['STATUS_BIN'], inplace=True)

print("\nMerged Dataset Shape:", final_df.shape)
print("\nMissing Values After Merge:")
print(final_df.isnull().sum())

# ============================================================
# STEP 10: ENCODE CATEGORICAL COLUMNS
# ============================================================
cat_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    final_df[col] = le.fit_transform(final_df[col])
    encoders[col] = le

print("\nFinal Dataset Head:")
print(final_df.head())
print("\nFinal Dataset Shape:", final_df.shape)

# Save cleaned dataset
final_df.to_csv('cleaned_data.csv', index=False)
print("\n✅ Cleaned data saved as cleaned_data.csv")