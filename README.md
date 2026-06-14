# AI-Based Hiring Prediction System

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-red?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## 📌 Project Overview

This project focuses on predicting whether a candidate is likely to be **Hired** or **Rejected** based on candidate information such as skills, experience, education, certifications, salary expectations, and project experience.

The project follows a complete machine learning workflow including data preprocessing, NLP-based feature engineering, handling class imbalance, model comparison, hyperparameter tuning, and deployment using Streamlit.

---

## 🚀 Live Demo

### 🌐 Streamlit Application

https://ai-based-hiring-system.streamlit.app

### 💻 GitHub Repository

https://github.com/Sanjayduduka45/Ai-Based-Hiring-System

---

## 🎯 Problem Statement

Recruiters often receive a large number of applications for a single role. Manually reviewing every profile can be time-consuming and inconsistent.

The goal of this project is to build a machine learning system capable of predicting hiring decisions using candidate profile information.

### Target Variable

| Class | Description |
| ----- | ----------- |
| 1     | Hire        |
| 0     | Reject      |

---

## 📂 Dataset Features

The dataset contains candidate information such as:

* Skills
* Experience (Years)
* Education
* Certifications
* Job Role
* Salary Expectation
* Projects Count
* Recruiter Decision (Target Variable)

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

* Missing value treatment
* Data cleaning
* Feature selection
* Train-test split

### 2️⃣ NLP Feature Engineering

The **Skills** column contains textual information.

To convert text into machine-readable numerical features:

* TF-IDF Vectorization was applied

### 3️⃣ Handling Class Imbalance

Dataset Distribution:

* Hire → 81.2%
* Reject → 18.8%

To improve minority class learning:

* SMOTE (Synthetic Minority Oversampling Technique) was used

### 4️⃣ Machine Learning Model Building

The following models were trained and compared:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost

### 5️⃣ Hyperparameter Tuning

RandomizedSearchCV with Cross Validation was used to optimize:

* XGBoost
* LightGBM
* CatBoost

---

## 📊 Model Performance

| Model               | Accuracy  | F1 Score   | ROC-AUC    |
| ------------------- | --------- | ---------- | ---------- |
| Logistic Regression | 95.0%     | 96.84%     | 99.68%     |
| Random Forest       | 90.0%     | 94.08%     | 96.43%     |
| XGBoost             | **98.0%** | **98.75%** | **99.64%** |
| LightGBM            | 95.5%     | 97.20%     | 99.24%     |
| CatBoost            | 96.0%     | 97.50%     | 99.29%     |

---

## 🏆 Best Performing Model

### XGBoost Classifier

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 98.0%  |
| Precision | 100%   |
| Recall    | 97.5%  |
| F1 Score  | 98.75% |
| ROC-AUC   | 99.64% |

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* LightGBM
* CatBoost
* Imbalanced-Learn
* Matplotlib
* Seaborn
* Streamlit

### NLP Techniques

* TF-IDF Vectorization

---

## 📈 Key Learnings

Through this project, I gained practical experience in:

* Data preprocessing and cleaning
* Natural Language Processing (TF-IDF)
* Handling imbalanced datasets using SMOTE
* Building machine learning pipelines
* Model evaluation and comparison
* Hyperparameter tuning
* Model deployment using Streamlit

---

## 📁 Project Structure

```text
Ai-Based-Hiring-System
│
├── app.py
├── hiring_prediction_model.pkl
├── requirements.txt
├── AI_Based_Hiring.ipynb
├── README.md
└── assets/
```

---

## 🔮 Future Improvements

* Resume PDF Parsing
* Candidate Ranking System
* Explainable AI using SHAP
* Interactive Recruiter Dashboard
* Integration with Applicant Tracking Systems (ATS)

---

## 🎓 Conclusion

This project demonstrates how Machine Learning and NLP can be used to support hiring decisions using candidate profile information. By combining TF-IDF feature extraction, SMOTE, model comparison, and hyperparameter tuning, the system achieved strong predictive performance.

Among all evaluated models, XGBoost delivered the best results with **98% Accuracy**, **98.75% F1 Score**, and **99.64% ROC-AUC**. The final model was successfully deployed using Streamlit, providing an interactive interface for predicting hiring outcomes.

This project strengthened my understanding of end-to-end machine learning workflows, including data preprocessing, feature engineering, model optimization, evaluation, and deployment.
