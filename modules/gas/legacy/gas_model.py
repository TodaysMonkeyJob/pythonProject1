# Research script kept for reference (thesis). Not used at runtime — see modules/gas/train.py.
# Restored from a PDF copy: indentation and wrapped lines fixed, page numbers removed.
# Reads data/gas/data_clean.xlsx (python -m modules.gas.data); the raw data.xls layout breaks transform_dataframe().
# Known issues (fixed in train.py): augmentation before the split leaks test rows into training,
# C = 0 rows get four different labels, no feature scaling, the 0.0 V column is dropped.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = 'data/gas/data_clean.xlsx'

ethanol = pd.read_excel(DATA_PATH, sheet_name="ethanol", header=None)
ammonia = pd.read_excel(DATA_PATH, sheet_name="ammonia", header=None)
acetone = pd.read_excel(DATA_PATH, sheet_name="acetone", header=None)
toluol = pd.read_excel(DATA_PATH, sheet_name="toluol", header=None)


def transform_dataframe(dataframe):
    new_columns = dataframe.iloc[-1]
    dataframe = dataframe.rename(columns=new_columns)
    dataframe = dataframe.set_index('C, %')
    dataframe = dataframe[:-1]
    return dataframe


ethanol = transform_dataframe(ethanol)
ammonia = transform_dataframe(ammonia)
acetone = transform_dataframe(acetone)
toluol = transform_dataframe(toluol)


def imputations(dataframe):
    df_columns = list(dataframe.columns)
    for i in range(len(df_columns)):
        try:
            dataframe[round((float(df_columns[i]) + float(df_columns[i+1]))/2, 2)] = \
                (dataframe[df_columns[i]] + dataframe[df_columns[i+1]]) / 2
        except Exception:
            pass
    sorted_columns = sorted(dataframe.columns, key=float)
    sorted_dataframe = dataframe[sorted_columns]
    return sorted_dataframe


ethanol = imputations(ethanol)
ammonia = imputations(ammonia)
acetone = imputations(acetone)
toluol = imputations(toluol)


def build_plot(df):
    x_size = df.shape[1]
    y_size = df.shape[0]
    x = np.arange(x_size)
    y = np.arange(y_size)
    X, Y = np.meshgrid(x, y)
    Z = df.values
    x_values = df.columns
    x_labels = x_values[::4]
    y_values = df.index
    y_labels = y_values[::5]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels, rotation=-20)
    ax.set_xlabel('V')
    ax.set_ylabel('С %')
    ax.set_zlabel('R')
    ax.view_init(15, 105)
    plt.show()


build_plot(ethanol)
build_plot(ammonia)
build_plot(acetone)
build_plot(toluol)


def calculate_average_sum(df):
    result = pd.DataFrame(columns=df.columns)
    for i in range(len(df) - 1):
        row_sum = (df.iloc[i] + df.iloc[i + 1]) / 2
        row_index = round((df.index[i] + df.index[i+1]) / 2, 2)
        result.loc[row_index] = row_sum
    return result


ethanol_df = calculate_average_sum(ethanol)
ethanol = pd.concat([ethanol, ethanol_df]).sort_index()
ammonia_df = calculate_average_sum(ammonia)
ammonia = pd.concat([ammonia, ammonia_df]).sort_index()
acetone_df = calculate_average_sum(acetone)
acetone = pd.concat([acetone, acetone_df]).sort_index()
toluol_df = calculate_average_sum(toluol)
toluol = pd.concat([toluol, toluol_df]).sort_index()

ethanol['Type'] = '0'
ammonia['Type'] = '1'
acetone['Type'] = '2'
toluol['Type'] = '3'

ethanol = ethanol.drop(ethanol.columns[0], axis=1)
ammonia = ammonia.drop(ammonia.columns[0], axis=1)
acetone = acetone.drop(acetone.columns[0], axis=1)
toluol = toluol.drop(toluol.columns[0], axis=1)

df_all = pd.concat([ethanol, ammonia, acetone, toluol])
df_split_1 = df_all.groupby(df_all.index < 5).get_group(True).reset_index(drop=True)
df_split_2 = df_all.groupby(df_all.index > 5).get_group(True).reset_index(drop=True)
df_all = df_all.reset_index(drop=True)
df = df_all.copy()
X = df.iloc[:, :-1]
y = df['Type']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_list = list()
model_name = ['Model SVM', 'Model GradientBoostingClassifier', 'Model KNeighborsClassifier',
              'Model DecisionTreeClassifier']

# Оптимізація параметрів моделі SVC
param_grid_svc = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.1, 1, 10, 100]}
grid_svc = GridSearchCV(SVC(), param_grid_svc, cv=10)
grid_svc.fit(X_train, y_train)
best_model_svc = grid_svc.best_estimator_
model_list.append(best_model_svc)

# Оптимізація параметрів моделі GradientBoostingClassifier
param_grid_grad = {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
grid_grad = GridSearchCV(GradientBoostingClassifier(), param_grid_grad, cv=5)
grid_grad.fit(X_train, y_train)
best_model_grad = grid_grad.best_estimator_
model_list.append(best_model_grad)

# Оптимізація параметрів моделі KNeighborsClassifier
param_grid_knc = {'n_neighbors': [5, 10, 15, 20]}
grid_knc = GridSearchCV(KNeighborsClassifier(), param_grid_knc, cv=10)
grid_knc.fit(X_train, y_train)
best_model_knc = grid_knc.best_estimator_
model_list.append(best_model_knc)

# Оптимізація параметрів моделі DecisionTreeClassifier
param_grid_dtc = {'max_depth': [None, 10, 20, 30]}
grid_dtc = GridSearchCV(DecisionTreeClassifier(), param_grid_dtc, cv=10)
grid_dtc.fit(X_train, y_train)
best_model_dtc = grid_dtc.best_estimator_
model_list.append(best_model_dtc)


def prediction_score(model, name):
    y_pred = model.predict(X_test)
    predictions = model.predict(X_test)
    predicted_materials = le.inverse_transform(predictions)
    test = list(map(int, y_test))
    predicted_materials = list(map(int, predicted_materials))
    actual, predicted = test[:25], predicted_materials[:25]
    print(name)
    print(classification_report(y_test, y_pred))
    print(f'Actual: {actual}', f'Predicted: {predicted}', "\n", sep="\n")


for i in range(len(model_list)):
    prediction_score(model_list[i], model_name[i])
