import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import numpy as np

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

# ===================== EDA =====================
print(df.head())
df.columns
print(df.info())
print(df.describe())
print(df.isnull().sum())

# checking duplicate rows
df.duplicated().sum()

# target variable distribution
df['HeartDisease'].value_counts().plot(kind='bar')

# function to plot histograms for numerical columns
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)

plotting('Age', 1)    
plotting('RestingBP', 2)    
plotting('Cholesterol', 3)
plotting('MaxHR', 4)

plt.tight_layout()

'''
This will plot histogram for numerical columns in the dataset.
After examining the plots:
- Many people have cholesterol level as 0 which is not possible
- Some people have resting blood pressure as 0 which is also not possible

So we will clean the data by replacing 0 values with mean.
'''

# ===================== DATA CLEANING =====================

# Calculate mean of Cholesterol excluding 0 values
ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()

# Replace 0 with mean and round to 2 decimals
df['Cholesterol'] = df['Cholesterol'].replace(0, ch_mean).round(2)

# Calculate mean of RestingBP excluding 0 values
resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()

# Replace 0 with mean and round to 2 decimals
df['RestingBP'] = df['RestingBP'].replace(0, resting_bp_mean).round(2)

# Verify the changes after cleaning
plotting('Age', 1)    
plotting('RestingBP', 2)    
plotting('Cholesterol', 3)
plotting('MaxHR', 4)

plt.tight_layout()

# automated EDA using sheryanalysis library
import sheryanalysis as sh  
sh.analyze(df)

# ===================== VISUAL ANALYSIS =====================

sns.countplot(x=df['sex'], hue=df['HeartDisease'])
plt.show()

sns.countplot(x=df['ChestPainType'], hue=df['HeartDisease'])
plt.show()

sns.countplot(x=df['FastingBS'], hue=df['HeartDisease'])
plt.show()

sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
plt.show()

sns.violinplot(x='HeartDisease', y='MaxHR', data=df)
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# ===================== FEATURE ENGINEERING =====================

'''
Feature engineering is the process of creating new features from existing data
to improve the performance of machine learning models.
Here we create medically meaningful features related to heart disease.
'''

# creating age groups
df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[0, 40, 60, 100],
    labels=['Young', 'Middle_Aged', 'Senior']
)

# categorizing cholesterol levels based on medical thresholds
df['CholesterolCategory'] = pd.cut(
    df['Cholesterol'],
    bins=[0, 200, 240, 600],
    labels=['Normal', 'Borderline', 'High']
)

# categorizing resting blood pressure
df['BP_Category'] = pd.cut(
    df['RestingBP'],
    bins=[0, 80, 120, 140, 300],
    labels=['Low', 'Normal', 'Pre_High', 'High']
)

# flag feature to identify patients with low max heart rate
df['Low_MaxHR'] = np.where(df['MaxHR'] < 100, 1, 0)

# derived feature to represent cardiac stress
df['Cardiac_Stress_Index'] = df['RestingBP'] / df['MaxHR']

print(df[['AgeGroup', 'CholesterolCategory', 'BP_Category',
          'Low_MaxHR', 'Cardiac_Stress_Index']].head())

# ===================== DATA ENCODING =====================

'''
Machine learning models cannot understand categorical (text) data.
So we convert categorical variables into numerical format using one-hot encoding.
'''

df_encode = pd.get_dummies(df, drop_first=True)
df_encode = df_encode.astype(float)

# ===================== DATA SCALING =====================

'''
Feature scaling is used to bring all numerical features
to the same scale so that no feature dominates the model.
'''

from sklearn.preprocessing import StandardScaler

numeric_cols = [
    'Age', 'RestingBP', 'Cholesterol',
    'MaxHR', 'Oldpeak', 'Cardiac_Stress_Index'
]

scaler = StandardScaler()
df_encode[numeric_cols] = scaler.fit_transform(df_encode[numeric_cols])

print(df_encode.head())
