# Uploading the necessary libraries and data
import matplotlib.pyplot as plt
import shap
import dalex
import yellowbrick
import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from dalex import Explainer
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import Image
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('diabetes.csv', sep=',', decimal=',')

data.info()
data.sample(10)

# Looking for missing data
data.isnull().sum()


# Convert BMI and DiabetesPedigreeFunction columns
data['BMI'] = pd.to_numeric(data['BMI'], errors='coerce')
data['DiabetesPedigreeFunction'] = pd.to_numeric(data['DiabetesPedigreeFunction'], errors='coerce')


data.dtypes

# Basic descriptive statistics
data.describe()

# Influence of individual variables on the forecast variable Outcome
sns.heatmap(data.corr(), cmap='Spectral',
center=0, vmin=-1, vmax=1)

max_corr_values = data.corr()['Outcome'].abs().drop('Outcome').nlargest(4)
max_corr_indices = max_corr_values.index

print(f"Największe korelacje ze zmienną 'Outcome':")
for idx in max_corr_indices:
    max_corr_value = data.corr()['Outcome'][idx]
    print(f"{idx}: {max_corr_value:.2f}")
    
min_corr_values = data.corr()['Outcome'].abs().drop('Outcome').nsmallest(4)
min_corr_indices = min_corr_values.index

print(f"Najmniejsze korelacje ze zmienną 'Outcome':")
for idx in min_corr_indices:
    min_corr_value = data.corr()['Outcome'][idx]
    print(f"{idx}: {min_corr_value:.2f}")


# Outliers
data_dropped_outliers = data[(data['Insulin'])<400]
plt.boxplot(data_dropped_outliers.values, labels=data_dropped_outliers.columns)
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Values')
plt.title('Box Plot of Features in Data')

plt.savefig('box_plot.png')
plt.show()

data = data_dropped_outliers

data['Outcome'].value_counts()

data_sorted = data.groupby('Outcome')
minSize = data_sorted.size().min()
data_balanced = data_sorted.apply(lambda x: x.sample(minSize, random_state=21)).reset_index(drop=True)
data_balanced['Outcome'].value_counts()

# Splitting the data into a teaching and test set
X = data.drop(columns=['Outcome'])
Y = data['Outcome']
cols = [k for k in data_balanced.columns if k != 'Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    data_balanced[cols],
    data_balanced['Outcome'],
    test_size=0.3,
    random_state=21
)
print((data_balanced[cols]))

# Standardization
scaler = StandardScaler()

scaler.fit(X_train)

# Scaling training and test data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculating the mean and standard deviation for scaled training data
mean_values = np.mean(X_train_scaled, axis=0)
std_values = np.std(X_train_scaled, axis=0)

print("Średnie dla przeskalowanych danych treningowych:")
print(mean_values)

print("\nOdchylenia standardowe dla przeskalowanych danych treningowych:")
print(std_values)

# KNN model
neighbors = np.arange(1, 20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_accuracy[i] = knn.score(X_train_scaled, y_train)
    test_accuracy[i] = knn.score(X_test_scaled, y_test)
    print(f"k: {k}, Test accuracy: {test_accuracy[i]}, Train accuracy: {train_accuracy[i]}")

plt.title('KNN')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# Error matrix and quality of KNN classification
chosen_k = 11
knn_scaled = KNeighborsClassifier(n_neighbors=chosen_k)
cm1 = ConfusionMatrix(knn_scaled)
cm1.fit(X_train_scaled, y_train)
accuracy = cm1.score(X_test_scaled, y_test)
cm1.show()

y_pred_scaled = knn_scaled.predict(X_test_scaled)
report = classification_report(y_test, y_pred_scaled)
print(report)

# Logistic regression model
model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)

conf_matrix = confusion_matrix(y_train, y_pred_train)
conf_matrix_with_labels = pd.DataFrame(conf_matrix, columns=['neg', 'pos'] , index=['neg', 'pos'] )
sns.heatmap(conf_matrix_with_labels, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()
print(classification_report(y_train, y_pred_train))

# ROC curve
y_pred_proba = model.predict_proba(X_train)[:, 1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)

# AUC (Area Under the Curve)
auc = roc_auc_score(y_train, y_pred_proba)

# ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Classification measures on learning and test sets
print("Zbiór uczący:")
report = classification_report(y_train, y_pred_train)
print(report)

y_pred_test = model.predict(X_test)

print("Zbiór testowy:")
report = classification_report(y_test, y_pred_test)
print(report)

# Decision tree
model3 = DecisionTreeClassifier(max_depth=4)
model3.fit(X_train, y_train)

y_pred_train = model3.predict(X_train)

conf_matrix = confusion_matrix(y_train, y_pred_train)
conf_matrix_with_labels = pd.DataFrame(conf_matrix, columns=['neg', 'pos'] , index=['neg', 'pos'] )
sns.heatmap(conf_matrix_with_labels, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix')
plt.show()


class_names_str = [str(class_name) for class_name in model3.classes_]
X_train_df = pd.DataFrame(X_train, columns=X.columns)

dot_data = export_graphviz(model3, out_file=None,
                           feature_names=list(X_train_df.columns),
                           class_names=class_names_str,
                           filled=True, rounded=True, special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())


print("Zbiór uczący:")
report = classification_report(y_train, y_pred_train)
print(report)


y_pred_test = model3.predict(X_test)

print("Zbiór testowy:")
report = classification_report(y_test, y_pred_test)
print(report)

# Random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred_train = rf_model.predict(X_train)

print("Zbiór treningowy:")
print(classification_report(y_train, y_pred_train))

conf_matrix_train = confusion_matrix(y_train, y_pred_train)
conf_matrix_with_labels_train = pd.DataFrame(conf_matrix_train, columns=['neg', 'pos'], index=['neg', 'pos'])
sns.heatmap(conf_matrix_with_labels_train, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix - Train Set')
plt.show()

y_pred_test = rf_model.predict(X_test)

print("Zbiór testowy:")
print(classification_report(y_test, y_pred_test))

conf_matrix_test = confusion_matrix(y_test, y_pred_test)
conf_matrix_with_labels_test = pd.DataFrame(conf_matrix_test, columns=['neg', 'pos'], index=['neg', 'pos'])
sns.heatmap(conf_matrix_with_labels_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix - Test Set')
plt.show()

constant_params = {
    'n_estimators': 100,
    'max_depth': None
}

param_names = ['n_estimators', 'max_features', 'max_depth']
param_values = [[50, 100, 200, 300, 400], ['sqrt', 'log2'], [10, 30,50,70,90,100, None]]

for param_name, param_vals in zip(param_names, param_values):
    for val in param_vals:
        params = constant_params.copy()
        params[param_name] = val

        rf = RandomForestClassifier(**params)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        print(f"{param_name}={val}:")
        print("Dokładność na zbiorze testowym:", accuracy_score(y_test, y_pred))
        print("Czułość:", recall_score(y_test, y_pred))
        print("Specyficzność:", precision_score(y_test, y_pred))
        print("-" * 50)


# Cross-validation of the model for the best parameter from random forests
best_rf = RandomForestClassifier(n_estimators=100, max_features="log2", max_depth=100)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Dokładność:", accuracy)
print("Czułość:", recall)
print("Precyzja:", precision)
print("-" * 50)

constant_params = {
    'n_estimators': 100,
    'max_depth': None
}

param_names = ['n_estimators', 'max_features', 'max_depth']
param_values = [[100, 200, 300, 400], ['sqrt', 'log2'], [10, 30, 50, 70, 90, 100, None]]

rf = RandomForestClassifier(**constant_params)

param_grid = {param_name: values for param_name, values in zip(param_names, param_values)}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Najlepsze parametry:", grid_search.best_params_)

best_rf = RandomForestClassifier(n_estimators=200, max_features="log2", max_depth=90)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

print("Dokładność:", accuracy_score(y_test, y_pred))
print("Czułość:", recall_score(y_test, y_pred))
print("Specyficzność:", precision_score(y_test, y_pred))
print("-" * 50)

# SHAP values
explainer = shap.Explainer(best_rf)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)