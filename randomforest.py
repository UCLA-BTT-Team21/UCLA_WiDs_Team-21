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
from sklearn.linear_model import LogisticRegression, RidgeClassifier,RidgeClassifierCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
from utils import get_feats, check_for_nulls
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

DATA_FOLDER = 'data/'
FIGURE_FOLDER = 'img/'
RESULT_FOLDER = 'results/'

train_file = 'data/train-edited.csv'
test_file = 'data/test-edited.csv'

# Load the train and test datasets using pandas
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

#NEW STUFF END
sub = pd.read_excel('data/SAMPLE_SUBMISSION.xlsx')
y = pd.read_excel(f"data/TRAIN/TRAINING_SOLUTIONS.xlsx")

train.set_index('participant_id',inplace=True)
test.set_index('participant_id',inplace=True)
targets = ['ADHD_Outcome','Sex_F']
features = test.columns

check_for_nulls(train)
check_for_nulls(test)
print(f'Train: {train.shape}, Test: {test.shape}')



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train, y[targets], test_size=0.30, random_state=42)

# Create a Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    # 'pca__n_components': [0.95, 0.90, 0.85, 100, 200],  # PCA components or variance threshold
    'rf__n_estimators': randint(100, 200),  # Number of trees
    'rf__max_depth': [None, 10, 20],        # Tree depth
    'rf__min_samples_split': randint(2, 5), # Minimum samples required to split
    'rf__min_samples_leaf': randint(1, 3),  # Minimum samples required at leaf
    'rf__bootstrap': [True, False]          # Bootstrap sampling
}

# Create the pipeline with PCA and Random Forest Classifier
pipeline = Pipeline([
    ('pca', PCA(0.95)),  # PCA step
    ('rf', rf_model)  # Random Forest step
])

# Set up RandomizedSearchCV with the pipeline
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, 
                                   n_iter=50, cv=3, n_jobs=-1, verbose=2, random_state=42)

# Fit the random search with the training data
random_search.fit(X_train, y_train)

# Make predictions with the best model
y_pred = random_search.best_estimator_.predict(X_test)

# Evaluate the model
print('F1 Score: ', f1_score(y_test, y_pred, average='micro'))

# Optionally: Check feature importance from the Random Forest part
print("Feature Importances:", random_search.best_estimator_.named_steps['rf'].feature_importances_)



y_pred = random_search.best_estimator_.predict(test)
sub['ADHD_Outcome'] = y_pred[:,0]
sub['Sex_F'] = y_pred[:,1]

sub.to_csv(f'{RESULT_FOLDER}results3.csv',index=False)