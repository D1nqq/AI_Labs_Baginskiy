import time
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Вхідний файл
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 5000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

X = np.array(X)
y = np.array(y)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.zeros((X.shape[0], X.shape[1]))

for i in range(X.shape[1]):
    le = preprocessing.LabelEncoder()
    X_encoded[:, i] = le.fit_transform(X[:, i])
    label_encoder.append(le)

X = X_encoded.astype(int)

# Нормалізація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Розділення даних
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Алгоритми для порівняння
models = [
    ('LR', LogisticRegression(max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(kernel='rbf', gamma='auto', cache_size=2000))  # Оптимізація для SVM
]

# Порівняння моделей
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    start_time = time.time()
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)  # Зменшити кількість фолдів
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)  # Паралельні обчислення
    end_time = time.time()
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f}) - Час: {end_time - start_time:.2f} сек")

# Порівняння алгоритмів
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів')
plt.xlabel('Алгоритм')
plt.ylabel('Точність')
plt.show()

# Оцінка найкращої моделі
start_time = time.time()
best_model = SVC(kernel='rbf', gamma='auto', cache_size=2000)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
end_time = time.time()

# Розрахунок метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Метрики для SVM:")
print(f"Точність: {accuracy:.4f}")
print(f"Прецизія: {precision:.4f}")
print(f"Повнота: {recall:.4f}")
print(f"F1 міра: {f1:.4f}")
print(f"Час для SVM: {end_time - start_time:.2f} сек")
