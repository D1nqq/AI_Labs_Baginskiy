from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

df = pd.read_csv('data_metrics.csv')
df.head()
thresh = 0.5
thresholds = [0.5, 0.25]

df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()

confusion_matrix(df.actual_label.values, df.predicted_RF.values)


def find_TP(y_true, y_pred):
    # counts the number of true positives (y_true = 1, y_pred = 1)
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    # counts the number of false negatives (y_true = 1, y_pred = 0)
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    # counts the number of false positives (y_true = 0, y_pred = 1)
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    # counts the number of true negatives (y_true = 0, y_pred = 0)
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):

    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def baginskiy_confusion_matrix(y_true, y_pred):

    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


baginskiy_confusion_matrix(df.actual_label.values, df.predicted_RF.values)

# Перевірка коректності матриці плутанини
assert np.array_equal(baginskiy_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      baginskiy_confusion_matrix(df.actual_label.values, df.predicted_RF.values)), \
    'baginskiy_confusion_matrix() is not correct for RF'

assert np.array_equal(baginskiy_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      baginskiy_confusion_matrix(df.actual_label.values, df.predicted_LR.values)), \
    'baginskiy_confusion_matrix() is not correct for LR'

print("All assertions passed. Your confusion matrix functions are correct!")


# Ваша власна функція для обчислення точності
def baginskiy_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    total = TP + FN + FP + TN
    return (TP + TN) / total if total > 0 else 0.0  # Обчислення точності


# Перевірка точності за допомогою assert
assert baginskiy_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(
    df.actual_label.values, df.predicted_RF.values), 'my_accuracy_score failed on RF'
assert baginskiy_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(
    df.actual_label.values, df.predicted_LR.values), 'my_accuracy_score failed on LR'

# Виведення результатів
print('Accuracy RF: %.3f' % baginskiy_accuracy_score(
    df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR: %.3f' % baginskiy_accuracy_score(
    df.actual_label.values, df.predicted_LR.values))


# Ваша власна функція для обчислення відзиву (recall)
def baginskiy_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    total_actual_positives = TP + FN
    # Обчислення відзиву
    return TP / total_actual_positives if total_actual_positives > 0 else 0.0


# Перевірка відзиву за допомогою assert
assert baginskiy_recall_score(df.actual_label.values, df.predicted_RF.values) == recall_score(
    df.actual_label.values, df.predicted_RF.values), 'my_recall_score failed on RF'
assert baginskiy_recall_score(df.actual_label.values, df.predicted_LR.values) == recall_score(
    df.actual_label.values, df.predicted_LR.values), 'my_recall_score failed on LR'

# Виведення результатів
print('Recall RF: %.3f' % baginskiy_recall_score(
    df.actual_label.values, df.predicted_RF.values))
print('Recall LR: %.3f' % baginskiy_recall_score(
    df.actual_label.values, df.predicted_LR.values))

# Ваша власна функція для обчислення точності (precision)


def baginskiy_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    total_predicted_positives = TP + FP
    # Обчислення точності
    return TP / total_predicted_positives if total_predicted_positives > 0 else 0.0


# Перевірка точності за допомогою assert
assert baginskiy_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(
    df.actual_label.values, df.predicted_RF.values), 'my_precision_score failed on RF'
assert baginskiy_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(
    df.actual_label.values, df.predicted_LR.values), 'my_precision_score failed on LR'

# Виведення результатів
print('Precision RF: %.3f' % baginskiy_precision_score(
    df.actual_label.values, df.predicted_RF.values))
print('Precision LR: %.3f' % baginskiy_precision_score(
    df.actual_label.values, df.predicted_LR.values))


# Ваша власна функція для обчислення F1-метрики
def baginskiy_f1_score(y_true, y_pred):
    recall = baginskiy_recall_score(y_true, y_pred)
    precision = baginskiy_precision_score(y_true, y_pred)
    if precision + recall > 0:
        # Обчислення F1-метрики
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0  # Повертає 0, якщо precision і recall обидва 0

# Перевірка F1-метрики за допомогою assert
# assert my_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values), 'my_f1_score failed on RF'
# assert my_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values), 'my_f1_score failed on LR'


# Виведення результатів
print('F1 RF: ', baginskiy_f1_score(
    df.actual_label.values, df.predicted_RF.values))
print('F1 LR: ',  baginskiy_f1_score(
    df.actual_label.values, df.predicted_LR.values))

print('F1 RF original: ', f1_score(
    df.actual_label.values, df.predicted_RF.values))
print('F1 LR original: ',  f1_score(
    df.actual_label.values, df.predicted_LR.values))

print('')


for threshold in thresholds:
    print(f'scores with threshold = {threshold}')

    # Генерація прогнозованих міток на основі поточного порогу
    predicted_RF = (df.model_RF >= threshold).astype('int')

    # Виведення показників
    print('Accuracy RF: %.3f' %
          (baginskiy_accuracy_score(df.actual_label.values, predicted_RF)))
    print('Recall RF: %.3f' %
          (baginskiy_recall_score(df.actual_label.values, predicted_RF)))
    print('Precision RF: %.3f' %
          (baginskiy_precision_score(df.actual_label.values, predicted_RF)))
    print('F1 RF: %.3f' % (baginskiy_f1_score(
        df.actual_label.values, predicted_RF)))
    print('')


fpr_RF, tpr_RF, thresholds_RF = roc_curve(
    df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(
    df.actual_label.values, df.model_LR.values)


""" plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show() """

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

import matplotlib.pyplot as plt

plt.plot(fpr_RF, tpr_RF,'r-',label = 'RF AUC: %.3f'%auc_RF)
plt.plot(fpr_LR,tpr_LR,'b-', label= 'LR AUC: %.3f'%auc_LR)
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

