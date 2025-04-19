# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Dataset Loading & Preparation: The Iris dataset is loaded using sklearn.datasets, converted into a DataFrame, and the target column (species labels) is added.

2.Feature and Target Splitting: Features (x) and labels (y) are separated, then split into training and testing sets using an 80-20 ratio.

3.Model Initialization & Training: A SGDClassifier is initialized with a maximum of 1000 iterations and trained using the training data.

4.Prediction: The model predicts species labels on the test set.

5.Evaluation: Model performance is evaluated using accuracy, confusion matrix, and classification report (precision, recall, f1-score).

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Saileshwaran Ganesan
RegisterNumber:  212224230237
*/
```
```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
```
```
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df
```
```
x = df.drop('target', axis=1)
y = df['target']
```
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
y_pred
```
```
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```
```
cmatrix=confusion_matrix(y_test,y_pred)
print("Confufion Matrix:")
print(cmatrix)
```
```
report=classification_report(y_test,y_pred)
print(report)
```

## Output:
Contents Of Datafile

![image](https://github.com/user-attachments/assets/d1dbba1f-e720-4605-96d0-5f72f2d7e1d0)

Predicted Values of Y

![image](https://github.com/user-attachments/assets/c18a9eaf-5b84-4f08-938c-790c3efa835a)

Accuracy

![image](https://github.com/user-attachments/assets/03b2c6ec-9179-4652-a27a-39f5b3308820)

Confusion Matrix

![image](https://github.com/user-attachments/assets/2fca57ce-e1e3-4433-ba3f-5ffdfa24c29f)

Classification Report:

![image](https://github.com/user-attachments/assets/e0795964-924f-48b4-abfe-e0a40c3ecf59)





## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
