# 🏠 House Price Prediction using Machine Learning (FAANG-Level Project)

## 🚀 Overview
This project is an end-to-end Machine Learning system that predicts house prices using advanced regression models and presents insights through a premium interactive dashboard UI.

It simulates how real-world platforms like Zillow, Housing.com, and banks estimate property values.

---

## 🎯 Problem Statement
Estimating house prices accurately is a complex problem influenced by multiple factors like location, size, quality, and amenities.

This project solves:
- Manual price guessing  
- Inconsistent pricing  
- Lack of data-driven insights  

By building:
A Machine Learning-powered prediction system.

---

## 💡 Key Features

### 🧠 Machine Learning
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor (Best Model)
- Model comparison & evaluation

---

### 📊 Data Analysis & Insights
- Correlation Heatmap
- Feature vs Price Analysis
- Business Insights (Avg, Min, Max price)
- Time Trend Analysis (YearBuilt vs Price)

---

### 🧠 Explainable AI (SHAP)
- Feature importance visualization  
- Model interpretability  
- Decision transparency  

---

### 🎨 Premium UI Dashboard
- Neon Black + Violet Theme  
- Multi-color interactive sliders  
- Animated background  
- Glassmorphism cards  
- Responsive layout  

---

### 📈 Advanced Visualizations
- Actual vs Predicted graph  
- Feature importance chart  
- Price distribution  
- Time-series trend  

---

### 📄 PDF Report Generation
- Download prediction results  
- Includes predicted price and model used  

---

## 🏗️ Project Architecture

Data → Cleaning → Feature Engineering → Model Training → Evaluation → Prediction → UI Dashboard

---

## 🛠️ Tech Stack

### Programming
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Plotly
- Streamlit
- ReportLab

---

## 📂 Project Structure

House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   ├── cleaned_train.csv
│   ├── X_processed.csv
│   └── y_processed.csv
│
├── models/
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── linear_regression.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── columns.pkl
│
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── predict.py
│
├── images/
├── outputs/
├── app.py
├── main.py
├── requirements.txt
└── README.md

---

## 📊 Dataset
Kaggle Dataset: House Prices - Advanced Regression Techniques  
~1460 training samples with 79+ features

---

## ⚙️ Installation

### 1. Clone Repository
git clone https://github.com/your-username/House-Price-Prediction-ML.git  
cd House-Price-Prediction-ML

### 2. Create Virtual Environment
python -m venv venv  

Activate:
venv\Scripts\activate   (Windows)

### 3. Install Dependencies
pip install -r requirements.txt  

---

## ▶️ Run the Project

### Run ML Pipeline
python main.py  

### Run Dashboard UI
streamlit run app.py  

---


## 📈 Model Performance

- Linear Regression → ~0.75 R²  
- Random Forest → ~0.88 R²  
- XGBoost → ~0.91 R²  

---

## 🧠 Business Impact

This system helps:
- Buyers → Fair price estimation  
- Sellers → Optimal listing price  
- Banks → Loan risk analysis  
- Investors → Market trend analysis  

---

## 🎯 Learning Outcomes

- End-to-end ML pipeline  
- Feature engineering  
- Model evaluation  
- Explainable AI (SHAP)  
- Dashboard development  
- Real-world problem solving  

---

## 🚀 Future Enhancements

- Deploy on Streamlit Cloud  
- Add more datasets  
- Deep Learning models  
- Location-based prediction  

---

## 🤝 Contribution
Feel free to fork this repository and improve it.

---

## Author
**Yuva Karthikeswar Dadisetty**

---

## ⭐ If you like this project
Give it a star on GitHub!