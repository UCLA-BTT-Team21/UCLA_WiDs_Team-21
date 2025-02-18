import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
import lightgbm as lgb, xgboost as xgb, catboost as cb
from gc import collect
import os
import matplotlib.pyplot as plt
import umap
from matplotlib.ticker import MaxNLocator
import scipy
import seaborn as sns
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PolynomialFeatures, MinMaxScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_curve, make_scorer
from sklearn.feature_selection import RFECV, SequentialFeatureSelector, SelectKBest
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTE
from utils import get_feats, check_for_nulls

DATA_FOLDER = 'data/'
FIGURE_FOLDER = 'img/'
RESULT_FOLDER = 'results/'

train = get_feats(mode='TRAIN')
test = get_feats(mode='TEST')
sub = pd.read_excel('data/SAMPLE_SUBMISSION.xlsx')
y = pd.read_excel(f"data/TRAIN/TRAINING_SOLUTIONS.xlsx")

train.set_index('participant_id', inplace=True)
test.set_index('participant_id', inplace=True)
targets = ['ADHD_Outcome', 'Sex_F']
features = test.columns

check_for_nulls(train)
check_for_nulls(test)

# Identify skewed features for log transformation
log_features = [f for f in features if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]

# Class Imbalance Handling (using SMOTE and class weights)
def get_class_weights(y_adhd, y_sex):
    weights = np.ones(len(y_adhd))
    # to adjust later for imbalance
    weights[(y_adhd == 1) & (y_sex == 1)] = 2  # Extra weight to female ADHD cases
    return weights

#smoting
smote = SMOTE(random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train.drop(targets, axis=1), y[targets], test_size=0.30, random_state=42)

# Define classifiers with potential class weights and other tuning
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'random_state': 42,
    'n_estimators': 200,
    'verbose': -1
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 200,
    'random_state': 42
}

catboost_params = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'Logloss',
    'random_state': 42
}

# Combine all the models into a multi-output classifier pipeline
models = [
    ('LGBM', lgb.LGBMClassifier(**lgb_params)),
    ('XGB', xgb.XGBClassifier(**xgb_params)),
    ('CatBoost', cb.CatBoostClassifier(**catboost_params, silent=True)),
    ('GradientBoosting', GradientBoostingClassifier(n_estimators=200, random_state=42))
]

# Create the pipeline
pipeline = make_pipeline(
    ColumnTransformer([('imputer', SimpleImputer(), features)], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas'),
    ColumnTransformer([('log', FunctionTransformer(np.log1p), log_features)], remainder='passthrough'),
    MinMaxScaler()
)

# Initialize MultiOutputClassifier
multi_model = MultiOutputClassifier(models[0][1])

# Fit model and predict
multi_model.fit(X_train, y_train)
y_pred = multi_model.predict(X_test)

# Evaluate with F1 score
print('F1 Score:', f1_score(y_test, y_pred, average='micro'))

# SMOTE application to the entire training set
X_res, y_res = smote.fit_resample(X_train, y_train)

# Train the models on the resampled data
multi_model.fit(X_res, y_res)
y_pred_resampled = multi_model.predict(X_test)

# Evaluate the resampled model
print('F1 Score (After SMOTE):', f1_score(y_test, y_pred_resampled, average='micro'))

# Feature Importance and Visualizations (similar to the previous code)
for model_name, model in models:
    plt.figure(figsize=(10, 6))
    if model_name == 'LGBM':
        lgb.plot_importance(model, max_num_features=20)
        plt.title(f'{model_name} Feature Importance')
    else:
        plt.title(f'{model_name} Model Importance (Approx.)')
    plt.tight_layout()
    plt.savefig(f"{FIGURE_FOLDER}{model_name}_Feature_Importance.png", dpi=300)
    plt.clf()

# UMAP Visualization
pipe = make_pipeline(SimpleImputer(), MinMaxScaler())
reducer = umap.UMAP()
x_scaler = pipe.fit_transform(train[features])
reducer.fit(x_scaler)
_, axs = plt.subplots(1, 2, figsize=(5, 3), constrained_layout=True)
embedding = reducer.transform(x_scaler)
for t, ax in zip(targets, axs.ravel()):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y[t], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    ax.set_title(f'{t}')
plt.suptitle('UMAP', fontsize=22)
plt.savefig(f"{FIGURE_FOLDER}UMAP.png", dpi=300)
plt.clf()

# Prepare final submission (same as before)
sub['ADHD_Outcome'] = y_pred[:, 0]
sub['Sex_F'] = y_pred[:, 1]
sub.to_csv(f'{RESULT_FOLDER}submission.csv', index=False)

#note: needs work