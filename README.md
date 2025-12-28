# Diabetes Prediction using Machine Learning

A machine learning model that predicts diabetes in patients based on medical features.

## Project Overview
- **Dataset:** 768 patients from Pima Indians Diabetes Database
- **Model:** Logistic Regression with StandardScaler
- **Accuracy:** 74.68%

## Key Features
- Data cleaning and preprocessing (handled 48% missing insulin data)
- Feature correlation analysis
- Model evaluation with confusion matrix
- Visualization of results

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Results
- **Accuracy:** 74.68%
- **Strongest Predictors:** Glucose (0.47 correlation), BMI (0.29 correlation)
- **Challenge:** Model has 21 false negatives (missed diabetic cases)

## How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
python diabetes_prediction.py
```

## 📝 Key Learnings
Data preprocessing took 70% of the project time. Real-world data requires extensive cleaning before modeling.

## 🔮 Future Improvements
- Try Random Forest for better accuracy
- Tune decision threshold to reduce false negatives
- Implement cross-validation
