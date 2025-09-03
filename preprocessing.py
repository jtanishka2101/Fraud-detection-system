# preprocessing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load Dataset
# -------------------------------
# Make sure creditcard.csv is in the same folder as this script
df = pd.read_csv("creditcard.csv")
print("Dataset loaded successfully!")
print(df.head())
print("\nDataset shape:", df.shape)

# -------------------------------
# 2. Handle Missing Values
# -------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())  # This dataset usually has no missing values

# -------------------------------
# 3. Normalize/Scale Features
# -------------------------------
# 'Amount' and 'Time' are not scaled in this dataset
scaler = StandardScaler()

df['Amount_Scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_Scaled'] = scaler.fit_transform(df[['Time']])

# Drop the old columns and keep the scaled ones
df = df.drop(['Amount', 'Time'], axis=1)

# -------------------------------
# 4. Exploratory Data Analysis
# -------------------------------
# Fraud vs Non-Fraud counts
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud (1) vs Non-Fraud (0) Transactions")
plt.show()

# Distribution of transaction amount
plt.figure(figsize=(6,4))
sns.histplot(df['Amount_Scaled'], bins=50, kde=True)
plt.title("Distribution of Transaction Amounts (Scaled)")
plt.show()

# -------------------------------
# 5. Save Cleaned Dataset
# -------------------------------
df.to_csv("cleaned_creditcard.csv", index=False)
print("\nCleaned dataset saved as cleaned_creditcard.csv")
