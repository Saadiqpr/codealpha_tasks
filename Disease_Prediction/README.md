# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a patient is likely to have heart disease using supervised machine learning algorithms.

The project was developed as part of the **CodeAlpha Machine Learning Internship** and demonstrates the complete machine learning workflow from data preprocessing to model evaluation and prediction.

---

## 🎯 Objectives

- Analyze the Heart Disease dataset.
- Perform data cleaning and preprocessing.
- Explore the dataset using visualizations.
- Train and compare multiple machine learning models.
- Select the best-performing model.
- Save the trained model for future predictions.

---

## 📂 Dataset

- **Dataset:** UCI Heart Disease Dataset
- **Source:** UCI Machine Learning Repository

The target variable was converted into a binary classification problem:

- **0** → No Heart Disease
- **1** → Heart Disease

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook (VS Code)

---

## 🤖 Machine Learning Models

The following algorithms were implemented:

- Gaussian Naive Bayes
- Logistic Regression
- Decision Tree
- Random Forest

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | **88.52%** |
| Gaussian Naive Bayes | **86.89%** |
| Logistic Regression | **86.89%** |
| Decision Tree | **78.69%** |

Random Forest achieved the highest accuracy and was selected as the final model.

---

## 📁 Project Structure

```text
Disease_Prediction/
│
├── Disease_Prediction.ipynb
├── README.md
├── requirements.txt
├── model/
│   └── best_heart_disease_model.pkl
├── dataset/
└── images/
```

---

## 🚀 How to Run

1. Clone the repository.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Open the notebook:

```text
Disease_Prediction.ipynb
```

4. Run all cells.

---

## 📈 Features

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Multiple ML Models
- Model Comparison
- Feature Importance Analysis
- Prediction Function
- Model Saving

---

## 🔮 Future Improvements

- Hyperparameter tuning
- Deploy as a Streamlit web application
- Use larger healthcare datasets
- Improve model explainability using SHAP or LIME

---

## 👨‍💻 Author

Developed as part of the **CodeAlpha Machine Learning Internship**.