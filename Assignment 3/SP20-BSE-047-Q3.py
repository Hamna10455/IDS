# 26-11-2023
# CSC461 – Assignment3 – Machine Learning
# Hamna Ashraf
# SP20-BSE-047
#Applying Random Forest classification algorithm (using Python) on the gender prediction 
#dataset with Monte Carlo cross-validation and Leave P-Out cross-validation. 
#Report F1 scores for both cross-validation strategies.


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, LeavePOut
from sklearn.metrics import make_scorer, f1_score
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("gender-prediction.csv")

df['beard'] = df['beard'].map({'yes': 1, 'no': 0})
df['hair_length'] = df['hair_length'].map({'bald': 0, 'short': 1, 'medium': 2, 'long': 3})
df['scarf'] = df['scarf'].map({'yes': 1, 'no': 0})
df['eye_color'] = df['eye_color'].astype('category').cat.codes  
X = df.drop('gender', axis=1)
y = df['gender']

rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

num_iterations = 5
monte_carlo_scores = cross_val_score(rf_classifier, X, y, cv=num_iterations, scoring=make_scorer(f1_score, average='weighted'))

p_out = 3
leave_p_out = LeavePOut(p=p_out)
leave_p_out_scores = cross_val_score(rf_classifier, X, y, cv=leave_p_out, scoring=make_scorer(f1_score, average='weighted'))

print(f"Monte Carlo Cross-Validation F1 Scores: {monte_carlo_scores.mean()}")
print(f"Leave P-Out Cross-Validation F1 Scores: {leave_p_out_scores.mean()}")
