import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
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

# Створення SVM-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0, max_iter=5000)) 

# Розділення даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Навчання класифікатора
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

# Обчислення показників якості
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)  
recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)  
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0) 

# Виведення результатів
print(f"Акуратність: {round(100 * accuracy, 2)}%")
print(f"Точність: {round(100 * precision, 2)}%")
print(f"Повнота: {round(100 * recall, 2)}%")


# Обчислення F-міри для SVM-класифікатора
f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=2) 

print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married',
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0

for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        # Спроба кодування значення
        if count < len(label_encoder):  # Перевірка, чи існує кодер
            try:
                input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
            except ValueError:
                print(f"Попередження: '{item}' не було у навчальних даних, використано умовне значення -1.")
                input_data_encoded[i] = -1  # Або використовувати інше стандартне значення
        else:
            print(f"Попередження: '{item}' не було у навчальних даних, використано умовне значення -1.")
            input_data_encoded[i] = -1  # Або використовувати інше стандартне значення
    count += 1

# Обрізаємо до 13 ознак, якщо потрібно
input_data_encoded = input_data_encoded[:13]  # Обрізаємо до 13 ознак

# Використання класифікатора для кодованої точки даних та виведення результату
predicted_class = classifier.predict([input_data_encoded])  # Додаємо квадратні дужки для 2D форми
print(label_encoder[-1].inverse_transform(predicted_class)[0])
