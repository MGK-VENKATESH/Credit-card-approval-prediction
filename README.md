
# 💳 Credit Card Approval Prediction App

This project is a machine learning-powered web application that predicts whether a credit card application is likely to be **approved or denied** based on user input. The app is built using **Python**, **scikit-learn**, and **Streamlit** and is intended for educational purposes.

---

## 📌 Project Overview

The goal of this project is to help visualize how machine learning models can be used in decision-making scenarios, such as credit risk assessment. The model is trained on a dataset containing applicants’ demographic, financial, and employment information, and uses a **Random Forest Classifier** to predict credit approval.

---

## 📊 Dataset

The dataset used in this project was sourced from Kaggle:

📎 **[Credit Card Approval Prediction Dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)**  
- `application_record.csv`: Contains personal and financial information of applicants  
- `credit_record.csv`: Contains historical credit behavior over time

> 💡 Credit goes to [Rikdifos on Kaggle](https://www.kaggle.com/rikdifos) for the dataset.

---

## 🚀 Features

- ✅ Predict credit card approval using personal & financial details
- 📊 User-friendly interface built with **Streamlit**
- 🧠 Machine learning model trained on real-world-style data
- 📦 Saves and loads models using `pickle`
- 📈 Displays approval confidence and key influencing factors

---

## 🚀 How to Run the Project

### 1. Clone the repository

bash code:
git clone https://github.com/MGK-VENKATESH/credit-approval-app.git
cd credit-approval-app

pip install -r requirements.txt

pip install pandas numpy scikit-learn streamlit

python credit.py

python credit.py

streamlit run app.py

---

## Model Performance
Algorithm: Random Forest Classifier

Accuracy on test set: ~89%

Key influencing features: income, employment history, education level, family status.

---
## Disclaimer
This app is for educational and demonstration purposes only. Real-world credit approvals involve far more complexity and regulatory checks.






