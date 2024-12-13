# Data Visualization:
# The code uses matplotlib to create scatter plots of the training data, distinguishing between two classes ('Not Purchased' and 'Purchased') and highlighting the support vectors.

# Model Training and Evaluation with Linear Kernel:
# An SVM model with a linear kernel is trained on the training data.
# The model's predictions on the test data are evaluated using a confusion matrix and accuracy score.

# Model Training and Evaluation with Different Kernels:
# The code iterates over a list of different kernel types ('rbf', 'poly', 'sigmoid').
# For each kernel, an SVM model is trained and evaluated on the test data, with the confusion matrix and accuracy score printed for each kernel.

# Model Training and Evaluation with Different Regularization Parameters:
# The code iterates over a list of different regularization parameter values (C).
# For each value of C, an SVM model with a linear kernel is trained and evaluated on the test data, and the accuracy scores are stored in a list.
# This approach allows for comparing the performance of SVM models with different kernels and regularization parameters to determine the best configuration for the given dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('Social_Network_Ads.csv')
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Not Purchased')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Purchased')
plt.title('Training Data')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


linear_svm = SVC(kernel='linear', C=1, random_state=0)
linear_svm.fit(X_train, y_train)
support_vectors = linear_svm.support_vectors_
plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='red', label='Not Purchased')
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='blue', label='Purchased')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='green', label='Support Vectors')
plt.title('SVM with Linear Kernel - Support Vectors')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

y_pred_linear = linear_svm.predict(X_test)
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print("Confusion Matrix (Linear Kernel):\n", conf_matrix_linear)
print("Accuracy (Linear Kernel):", accuracy_linear)

kernels = ['rbf', 'poly', 'sigmoid']
for kernel in kernels:
    svm_model = SVC(kernel=kernel, C=1, random_state=0)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nKernel: {kernel.capitalize()}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Accuracy:", accuracy)

c_values = [0.1, 1, 10, 100, 1000]
accuracy_scores = []

for c in c_values:
    svm_model = SVC(kernel='linear', C=c, random_state=0)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(c_values, accuracy_scores, marker='o')
plt.xlabel('C value (log scale)')
plt.ylabel('Accuracy')
plt.title('Effect of C value on SVM Accuracy (Linear Kernel)')
plt.show()
