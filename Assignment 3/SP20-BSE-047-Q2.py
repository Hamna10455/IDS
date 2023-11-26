# 26-11-2023
# CSC461 – Assignment3 – Machine Learning
# Hamna Ashraf
# SP20-BSE-047
#Applying Logistic Regression, Support Vector Machines, and Multilayer Perceptron classification algorithms (using Python) 
#on the gender prediction dataset with 2/3 train and 1/3 test split ratio.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from google.colab import files
uploaded = files.upload()

import io
df = pd.read_csv(io.StringIO(uploaded['gender-prediction.csv'].decode('utf-8')))

print(df.head())

X = df.drop('gender', axis=1)
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

y_pred = logreg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)

X_excluded = X.drop(['Height', 'Weight'], axis=1)

X_train_excluded, X_test_excluded, y_train_excluded, y_test_excluded = train_test_split(X_excluded, y, test_size=0.2, random_state=42)

lr_model_excluded = LogisticRegression()
lr_model_excluded.fit(X_train_excluded, y_train_excluded)
lr_predictions_excluded = lr_model_excluded.predict(X_test_excluded)
lr_accuracy_excluded = accuracy_score(y_test_excluded, lr_predictions_excluded)

svm_model_excluded = SVC()
svm_model_excluded.fit(X_train_excluded, y_train_excluded)
svm_predictions_excluded = svm_model_excluded.predict(X_test_excluded)
svm_accuracy_excluded = accuracy_score(y_test_excluded, svm_predictions_excluded)

mlp_model_excluded = MLPClassifier()
mlp_model_excluded.fit(X_train_excluded, y_train_excluded)
mlp_predictions_excluded = mlp_model_excluded.predict(X_test_excluded)
mlp_accuracy_excluded = accuracy_score(y_test_excluded, mlp_predictions_excluded)

print("\nTask 6: Results after excluding 'Height' and 'Weight':")
print("Logistic Regression Accuracy:", lr_accuracy_excluded)
print("Support Vector Machines Accuracy:", svm_accuracy_excluded)
print("Multilayer Perceptron Accuracy:", mlp_accuracy_excluded)

incorrect_lr_excluded = (y_test_excluded != lr_predictions_excluded).sum()
incorrect_svm_excluded = (y_test_excluded != svm_predictions_excluded).sum()
incorrect_mlp_excluded = (y_test_excluded != mlp_predictions_excluded).sum()

print("\nTask 6: Number of instances incorrectly classified without 'Height' and 'Weight':")
print("Logistic Regression:", incorrect_lr_excluded)
print("Support Vector Machines:", incorrect_svm_excluded)
print("Multilayer Perceptron:", incorrect_mlp_excluded)
