import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve

import warnings
warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 1000)
pd.set_option("display.expand_frame_repr", False)

#It is desired to develop a machine learning model that can predict whether people have diabetes when their characteristics are specified.
#You are expected to do data analysis and feature engineering steps required before developing model.

# Dataset

#The dataset is part of the large dataset held at the National Institutes # of Diabetes-Digestive-Kidney Diseases in the USA.
#Data used for diabetes research On Pima Indian women aged 21 and over living in Phoenix, the 5th largest city in the State of Arizona in the USA.
#The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 0 indicates negative.

#Pregnancies: Number of pregnancies
#Glucose: 2-hour plasma glucose concentration in the oral glucose tolerance test
#Blood Pressure: Blood Pressure (Small blood pressure) (mm Hg)
#SkinThickness
#Insulin: 2-hour serum insulin (mu U/ml)
#DiabetesPedigreeFunction: Function (2 hour plasma glucose concentration in oral glucose tolerance test)
#BMI: Body Mass Index
#Age
#Outcome: Have the disease (1) or not (0)

# Data Preprocessing
def load():
    data = pd.read_csv(r"C:\Users\sermi\PycharmProjects\pythonProject4\DIABETES\diabetes.csv")
    return data

df = load()
df.head()
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

#Analyze numerical and categorical variables.
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Categorical variable analysis
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome",plot=True)

# Numerical variable analysis
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)
    plt.pause(5)

# Target variable analysis
def numerical_vs_target(dataframe, target, numerical_col):
    temp_df = dataframe.copy()
    for col in numerical_col:
        print(pd.DataFrame({col : temp_df.groupby(target)[col].mean(),
                            "Count": temp_df.groupby(target)[col].count()}), end="\n\n\n")

numerical_vs_target(df, "Outcome", num_cols)

# Correlation analysis

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)

corrMatrix = df.corr()
corrMatrix["Outcome"].sort_values(ascending=False)

# Base Model

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
#0.77
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
#0.706
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
#0.59
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
#0.64
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
#0.75

# FEATURE ENGINEERING

# Missing Values
# There are no missing observations in the data set, but Glucose, Insulin etc.
# Values containing a value of 0 may represent the missing value. For example; no one's glucose or insulin value will be 0.
na_columns = [col for col in df.columns if (df[col].min()==0 and col not in ["Pregnancies", "Outcome"])]

for col in na_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()
def missing_values_table(dataframe, na_name=False):
    nan_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[nan_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[nan_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 3)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return nan_columns, missing_df

nan_columns, missing_df = missing_values_table(df, na_name=True)

def missing_vs_target(dataframe, target, nan_columns):
    temp_df = dataframe.copy()
    for col in nan_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)

for col in na_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

# Outliers
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(f'{col} : {outlier_thresholds(df, col)}')

for col in num_cols:
    print(f'{col} : {check_outlier(df, col)}')

for col in df.columns:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(f'{col} : {check_outlier(df, col)}')

# New features

df.loc[(df['Age'] < 21), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 21) & (df['Age'] < 60), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 60), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['BMI'] < 18.5), 'NEW_BMI'] = 'low_bmi'
df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 24.9), 'NEW_BMI'] = 'normal_bmi'
df.loc[(df['BMI'] > 25) & (df['BMI'] < 29.9), 'NEW_BMI'] = 'overweight_bmi'
df.loc[(df['BMI'] >= 30), 'NEW_BMI'] = 'obese_bmi'

df.loc[(df['Insulin'] < 167), 'NEW_INSULIN'] = 'normal_ins'
df.loc[(df['Insulin'] >= 167) & (df['Insulin'] < 200), 'NEW_INSULIN'] = 'prediabet_ins'
df.loc[(df['Insulin'] >= 200), 'NEW_INSULIN'] = 'abnormal_ins'

df.loc[(df['BloodPressure'] < 80), 'NEW_BP'] = 'normal_bp'
df.loc[(df['BloodPressure'] >= 80) & (df['BloodPressure'] <=89), 'NEW_BP'] = 'high_bp'
df.loc[(df['BloodPressure'] > 90), 'NEW_BP'] = 'hyper_bp'

df.loc[(df['Pregnancies'] == 0), 'NEW_PREGNANCY_CAT'] = 'no_pregnancy'
df.loc[(df['Pregnancies'] == 1), 'NEW_PREGNANCY_CAT'] = 'one_pregnancy'
df.loc[(df['Pregnancies'] > 1), 'NEW_PREGNANCY_CAT'] = 'multi_pregnancy'

df.loc[(df['Glucose'] >= 170), 'NEW_GLUCOSE_CAT'] = 'dangerous'
df.loc[(df['Glucose'] >= 105) & (df['Glucose'] < 170), 'NEW_GLUCOSE_CAT'] = 'risky'
df.loc[(df['Glucose'] < 105) & (df['Glucose'] > 70), 'NEW_GLUCOSE_CAT'] = 'normal'
df.loc[(df['Glucose'] <= 70), 'NEW_GLUCOSE_CAT'] = 'low'

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["NEW_SKINTHICKNESS_BMI"] = df["SkinThickness"] / df["BMI"]
df["NEW_AGE_DPEDIGREE"] = df["Age"] / df["DiabetesPedigreeFunction"]
df["NEW_GLUCOSE_BLOODPRESSURE"] = (df["BloodPressure"] * df["Glucose"])/100

# LabelEncoding

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df, col)

# One-Hot Encoding
ohe_cols = [col for col in df.columns if (10 >= df[col].nunique() > 2 and col not in ["Outcome"])]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols, drop_first=True)

# StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

# Final Model

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
#0.77
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
#0.68
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
#0.67
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
#0.68
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
#0.75
def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[1:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    # plt.savefig('importances-01.png')
    plt.show()

plot_importance(rf_model, X)