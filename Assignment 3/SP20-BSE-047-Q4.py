# 26-11-2023
# CSC461 – Assignment3 – Machine Learning
# Hamna Ashraf
# SP20-BSE-047
#Evaluating the trained model using the newly added 10 test instances. 
#Reporting accuracy, precision, and recall scores.



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from google.colab import files

uploaded = files.upload()

df = pd.read_csv("gender-prediction.csv")

print("Dataset:")
print(df)

X = df.drop('gender', axis=1)
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
