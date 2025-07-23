# -----------------------------
# 📌 TASK 1: DATA QUALITY ASSESSMENT
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("ecommerce_customers_large.csv")

print("🔍 First 5 rows:\n", df.head())
print("\n📏 Shape:", df.shape)
print("\n💾 Memory usage:")
print(df.info(memory_usage='deep'))
print("\n🔡 Data types:\n", df.dtypes)
print("\n❌ Missing values per column:\n", df.isnull().sum())
print("\n📄 Duplicate rows:", df.duplicated().sum())
print("\n📊 Numeric summary:\n", df.describe())
print("\n📝 Categorical summary:\n", df.describe(include='object'))

plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='YlGnBu')
plt.title('Missing Values Heatmap')
plt.show()


# -----------------------------
# 📌 TASK 2.1: HANDLE MISSING VALUES
# -----------------------------

# Check % missing
missing_percent = df.isnull().mean() * 100
print("\n% Missing:\n", missing_percent)

# Correct column names!
if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'AnnualIncome' in df.columns:
    df['AnnualIncome'] = df['AnnualIncome'].fillna(df['AnnualIncome'].mean())

# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Fill categorical/text columns with mode
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n✅ Missing values handled.")


# -----------------------------
# 📌 TASK 2.2: REMOVE DUPLICATES
# -----------------------------

print("\n🔍 Exact Duplicates:\n", df[df.duplicated()])
df = df.drop_duplicates()
print("\n✅ Duplicates removed. Shape:", df.shape)


# -----------------------------
# 📌 TASK 2.3: FIX DATA TYPES
# -----------------------------

if 'registration_date' in df.columns:
    df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')

if 'last_order_date' in df.columns:
    df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')

if 'is_premium' in df.columns:
    df['is_premium'] = df['is_premium'].astype(bool)

if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].str.lower().str.strip()

print("\n✅ Data types fixed.\n", df.dtypes)


# -----------------------------
# 📌 TASK 2.4: CLEAN TEXT DATA
# -----------------------------

text_cols = ['first_name', 'last_name', 'email', 'phone']
present_cols = [col for col in text_cols if col in df.columns]

for col in present_cols:
    if col in ['first_name', 'last_name']:
        df[col] = df[col].str.strip().str.title()
    elif col == 'email':
        df[col] = df[col].str.strip().str.lower()
    elif col == 'phone':
        df[col] = df[col].str.strip().str.replace(" ", "")

if present_cols:
    print("\n✅ Text cleaned.\n", df[present_cols].head())
else:
    print("\n⚠️ No text columns found to clean.")


# -----------------------------
# 📌 TASK 3: OUTLIERS
# -----------------------------

# Use correct column names
num_cols = ['Age', 'AnnualIncome']
if 'TotalSpent' in df.columns:
    num_cols.append('TotalSpent')
if 'OrdersCount' in df.columns:
    num_cols.append('OrdersCount')

for col in num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# IQR for AnnualIncome
Q1 = df['AnnualIncome'].quantile(0.25)
Q3 = df['AnnualIncome'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df['AnnualIncome'] >= lower) & (df['AnnualIncome'] <= upper)]

print("\n✅ Outliers treated. New shape:", df.shape)


# -----------------------------
# 📌 TASK 4: FEATURE ENGINEERING
# -----------------------------

# Age groups
df['age_group'] = pd.cut(df['Age'],
                         bins=[0, 18, 30, 45, 60, 100],
                         labels=['Teen', 'Young Adult', 'Adult', 'Mid Age', 'Senior'])

# Income bracket
df['income_bracket'] = pd.cut(df['AnnualIncome'],
                              bins=[0, 40000, 80000, 120000, 200000],
                              labels=['Low', 'Medium', 'High', 'Very High'])

print("\n✅ Feature engineering done.\n", df[['age_group', 'income_bracket']].head())
