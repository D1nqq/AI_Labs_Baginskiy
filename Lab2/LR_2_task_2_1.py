import time
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue

        data = line.strip().split(', ')

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])  # Додамо всі елементи, крім останнього, до X
            y.append(0)  # Мітка класу <=50K
            count_class1 += 1
        elif data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)  # Мітка класу >50K
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)
y = np.array(y)

# Перевірка на наявність даних
if X.size == 0:
    raise ValueError("Помилка: Дані не були зчитані коректно. Перевірте формат файлу або умови зчитування.")

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.zeros((X.shape[0], X.shape[1]), dtype=object)  # Задаємо правильну форму

for i in range(X.shape[1]):
    if np.issubdtype(X[:, i].dtype, np.number):  # Перевірка на числові дані
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X = X_encoded[:, :-1].astype(int)  
y = X_encoded[:, -1].astype(int)  

# Нормалізація даних
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Таймер для оцінки швидкості
start_time = time.time()

# Розділення даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Поліноміальне ядро з degree=8
classifier = SVC(kernel='poly', degree=8, random_state=0)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Оцінка часу навчання
train_time = time.time() - start_time

# Таймер для передбачення
start_time = time.time()
y_test_pred = classifier.predict(X_test)
predict_time = time.time() - start_time

# Обчислення показників якості
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"Акуратність: {round(100 * accuracy, 2)}%")
print(f"Точність: {round(100 * precision, 2)}%")
print(f"Повнота: {round(100 * recall, 2)}%")
print(f"F1 міра: {round(100 * f1, 2)}%")
print(f"Час навчання: {train_time:.4f} секунд")
print(f"Час передбачення: {predict_time:.4f} секунд")
