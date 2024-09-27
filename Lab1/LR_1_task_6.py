import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Завантаження даних
data = np.loadtxt('data_multivar_nb.txt', delimiter=',')

# Розбиття на ознаки і мітки
X = data[:, :-1]  # ознаки
y = data[:, -1]   # мітки

# Розділення на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Створення та навчання моделі SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred_svm = svm_model.predict(X_test)

# Розрахунок показників для SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
f1_svm = f1_score(y_test, y_pred_svm, average='macro')

print(f"SVM - Accuracy: {accuracy_svm:.3f}, Precision: {precision_svm:.3f}, Recall: {recall_svm:.3f}, F1 Score: {f1_svm:.3f}")
# Створення та навчання моделі наївного байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred_nb = nb_model.predict(X_test)

# Розрахунок показників для наївного Байєса
accuracy_nb = accuracy_score(y_test, y_pred_nb)
precision_nb = precision_score(y_test, y_pred_nb, average='macro')
recall_nb = recall_score(y_test, y_pred_nb, average='macro')
f1_nb = f1_score(y_test, y_pred_nb, average='macro')

print(f"Naive Bayes - Accuracy: {accuracy_nb:.3f}, Precision: {precision_nb:.3f}, Recall: {recall_nb:.3f}, F1 Score: {f1_nb:.3f}")
