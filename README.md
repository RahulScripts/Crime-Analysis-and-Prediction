---

# 📊 Crime Analysis and Prediction using Machine Learning

## 📌 Project Overview
This project focuses on analyzing crime data and predicting crime risk levels using machine learning regression models. By leveraging **Poisson Regression**, **Random Forest Regressor**, and **XGBoost Regressor**, the model predicts crime counts based on spatial and temporal features such as `BLOCK` and `SHIFT`. The goal is to help understand crime patterns and support decision-making in resource allocation and law enforcement strategies.

---

## 🚀 Features
✅ Crime data preprocessing and feature engineering  
✅ Model training with three regression algorithms  
✅ Model evaluation using RMSE and residual analysis  
✅ Feature importance analysis for Random Forest and XGBoost  
✅ Model comparison and best model selection  
✅ Joblib model export for future use  
✅ Predictions generated using the best model

---

## 🧠 Technologies Used
- Python 3
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- Joblib

---

## 📂 Dataset
The dataset used contains crime incident records with features like `BLOCK` and `SHIFT`. It is processed to generate crime counts for prediction.

**Note:** Dataset name - `Crime_Incidents_in_2024.csv`

---

## 🏗️ Project Workflow
1. Data Preprocessing (grouping, encoding)
2. Train-test split
3. Model training:
   - Poisson Regression
   - Random Forest Regressor
   - XGBoost Regressor
4. Performance Evaluation (RMSE, residuals)
5. Feature Importance Visualization
6. Model Recommendation
7. Model Export
8. Predictions using the best model

---

## 📈 Results
- **Best Model:** Selected based on lowest RMSE
- **Evaluation Metrics:** RMSE and Standard Deviation of residuals
- **Feature Importance:** Identified key predictors affecting crime risk

---

## 📥 Installation and Usage
1. Clone the repository:
```
git clone 
cd 
```
2. Install required libraries:
```
pip install -r requirements.txt
```
3. Run the Jupyter Notebook:
```
jupyter notebook Crime_Analysis_and_Prediction_final.ipynb
```

4. (Optional) Load the saved models:
```
import joblib
model = joblib.load('xgboost_model.joblib')
```

---

## 📚 Future Scope
- Include more granular data (weather, demographics)
- Implement geospatial mapping
- Real-time crime data integration
- Develop a web or mobile app interface
- Test deep learning models for enhanced accuracy

---

## 🤝 Acknowledgments
- Python open-source community
- Libraries like Pandas, Scikit-learn, XGBoost
- Detailed documentation that guided model implementation

---

## 📜 License
This project is licensed under the MIT License.

---

## 💻 Author
**Rahul Halli**

---
