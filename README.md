# Student Performance Prediction using KNN

A machine learning project that predicts whether a student's exam performance will be **Poor, Average, or Excellent** from study habits and background factors.

## Problem
Given details like attendance, hours studied, and previous scores, can we classify how well a student will perform in the exam?

## Dataset
- **Student Performance Factors** dataset (6,607 students, 20 columns)
- Target: `Exam_Score`, split into 3 performance classes using quantiles (`pd.qcut`)
- Missing values (Teacher_Quality, Parental_Education_Level, Distance_from_Home) filled with the most frequent value (mode)

## Approach
1. **Exploration**: exam score distribution, attendance and hours studied vs score, correlation heatmap, gender and motivation counts
2. **Preprocessing**: label encoding for categorical features, train/test split (80/20, stratified), standard scaling
3. **Feature selection**: Random Forest feature importance, kept the top 8 features:
   Attendance, Hours_Studied, Previous_Scores, Tutoring_Sessions, Sleep_Hours, Physical_Activity, Access_to_Resources, Parental_Involvement
4. **Model**: K-Nearest Neighbors, first with K=5 (76.40% accuracy), then tuned K from 1 to 75
5. **Evaluation**: accuracy, confusion matrix, precision / recall / F1 per class
6. **Prediction**: a small example that predicts performance for a new student

## Results
| Model | Accuracy |
|---|---|
| KNN (K = 5) | 76.40% |
| KNN (K = 40, tuned) | 78.37% |

Attendance and hours studied were the two most important features.

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Google Colab

## How to Run
1. Download the dataset and place it where the notebook can read it (update the path in the data loading cell)
2. Open the notebook in Jupyter or Google Colab
3. Run all cells from top to bottom

## Future Improvements
- Compare with Random Forest, Logistic Regression, and SVM
- Use cross-validation to choose K instead of the test set
- Deploy as a small web app with Streamlit

## Author
Kushal Pokhrel 
