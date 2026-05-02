# ❤️ Heart Disease EDA & Feature Engineering

A complete Exploratory Data Analysis (EDA), data cleaning, 
and preprocessing pipeline on a Heart Disease dataset to 
understand factors affecting heart disease risk.

---

## 📌 Problem Statement
Analyzing medical attributes like age, cholesterol, blood pressure, 
and heart rate to identify key risk factors associated with 
heart disease using statistical and visual analysis.

---

## 📊 Dataset
- **File:** heart.csv
- **Domain:** Medical / Cardiology
- **Target Variable:** HeartDisease (0 = No, 1 = Yes)
- **Features:** Age, Sex, ChestPainType, RestingBP, Cholesterol, 
FastingBS, MaxHR, Oldpeak and more

---

## 🔧 What This Project Covers

### ✅ EDA
- Dataset overview (shape, info, describe)
- Missing value and duplicate analysis
- Target variable distribution
- Distribution plots for Age, RestingBP, Cholesterol, MaxHR
- Automated EDA using SheryAnalysis library

### ✅ Data Cleaning
- Detected medically impossible 0 values in Cholesterol and RestingBP
- Replaced 0 values with column mean (excluding zeros)
- Before vs After cleaning visualization

### ✅ Visual Analysis
- Heart disease by Sex (countplot)
- Heart disease by ChestPainType
- Heart disease by FastingBS
- Cholesterol vs HeartDisease (boxplot)
- MaxHR vs HeartDisease (violinplot)
- Correlation heatmap

### ✅ Feature Engineering
- AgeGroup (Young / Middle_Aged / Senior)
- CholesterolCategory (Normal / Borderline / High)
- BP_Category (Low / Normal / Pre_High / High)
- Low_MaxHR flag (MaxHR < 100)
- Cardiac_Stress_Index (RestingBP / MaxHR)

### ✅ Encoding & Scaling
- One-Hot Encoding for all categorical features
- Standard Scaling for numerical features

---

## 📈 Key Findings
- Smokers and senior age group show higher heart disease risk
- High cholesterol and low MaxHR are strong indicators
- ASY chest pain type is most associated with heart disease
- Cardiac Stress Index is a useful derived feature

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (StandardScaler)
- SheryAnalysis (automated EDA)

---

## 📁 Project Structure
```
├── heart_disease_eda.py    # Main EDA script
├── heart.csv               # Dataset
└── README.md
```

---

## 🚀 How to Run
```bash
git clone https://github.com/shiva-18-cs/heart-disease-eda
cd heart-disease-eda
pip install pandas numpy matplotlib seaborn scikit-learn sheryanalysis
python heart_disease_eda.py
```

---

## 📚 Concepts Covered
- Exploratory Data Analysis
- Medical domain data cleaning
- Feature Engineering with domain knowledge
- One-Hot Encoding (drop_first=True)
- Standard Scaling
- Derived and flag features