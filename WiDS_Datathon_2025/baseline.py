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

DATA_FOLDER = 'data/'
FIGURE_FOLDER = 'img/'
RESULT_FOLDER = 'results/'

train=get_feats(mode='TRAIN')
test=get_feats(mode='TEST')
sub = pd.read_excel('data/SAMPLE_SUBMISSION.xlsx')
y = pd.read_excel(f"data/TRAIN/TRAINING_SOLUTIONS.xlsx")

train.set_index('participant_id',inplace=True)
test.set_index('participant_id',inplace=True)
targets = ['ADHD_Outcome','Sex_F']
features = test.columns

check_for_nulls(train)
check_for_nulls(test)
print(f'Train: {train.shape}, Test: {test.shape}')

fig, axs = plt.subplots(1,2, figsize=(5,3))
for col, ax in zip(y.drop('participant_id',axis=1), axs):    
    counts = y[col].value_counts()
    ax.pie(counts, labels=counts.index, 
           autopct='%1.1f%%', 
           startangle=90)
    ax.set_title(f'{col}')
plt.savefig(f"{FIGURE_FOLDER}Y_Pie_Charts.png", dpi=300, bbox_inches="tight")
plt.clf()

log_features = [f for f in features if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]

X_train, X_test, y_train, y_test = train_test_split(train.drop(targets,axis=1), y[targets], test_size=0.30, random_state=42)
model = MultiOutputClassifier(make_pipeline(ColumnTransformer([('imputer',SimpleImputer(),features)],
                                               remainder='passthrough',
                                               verbose_feature_names_out=False).set_output(transform='pandas'),
                                            ColumnTransformer([('log', 
                                                 FunctionTransformer(np.log1p), log_features)],
                                                 remainder='passthrough'),
                                            MinMaxScaler(),    
                                            RidgeClassifier(alpha=100)))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('f1: ', f1_score(y_test,y_pred,average='micro'))

pca = make_pipeline(SimpleImputer(),StandardScaler(),PCA())
pca.fit(train[test.columns])
plt.figure(figsize=(7,5))
plt.plot(pca[-1].explained_variance_ratio_.cumsum())
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.title('Principal Components Analysis')
plt.xlabel('component#')
plt.ylabel('explained variance ratio')
plt.yticks([0,0.5,0.85,0.90,0.95,1])
plt.xticks(range(0,1300,100))
plt.grid()
plt.savefig(f"{FIGURE_FOLDER}PCA.png", dpi=300)
plt.clf()

pipe = make_pipeline(SimpleImputer(),MinMaxScaler())
reducer = umap.UMAP()
x_scaler = pipe.fit_transform(train[features])
reducer.fit(x_scaler)
_, axs = plt.subplots(1,2, figsize=(5,3), constrained_layout=True)
embedding = reducer.transform(x_scaler)
for t,ax in zip(targets,axs.ravel()):    
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y[t], cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    ax.set_title(f'{t}')
plt.suptitle('UMAP',fontsize=22)
plt.savefig(f"{FIGURE_FOLDER}UMAP.png", dpi=300)
plt.clf()

model = MultiOutputClassifier(make_pipeline(ColumnTransformer([('imputer',SimpleImputer(),features)],
                                               remainder='passthrough',
                                               verbose_feature_names_out=False).set_output(transform='pandas'),
                                              ColumnTransformer([('log', 
                                                 FunctionTransformer(np.log1p), log_features)],
                                                 remainder='passthrough'),
                                            MinMaxScaler(),  
                                            PCA(1087),
                                            RidgeClassifier(alpha=100)))
model.fit(train.drop(targets,axis=1), y.drop('participant_id',axis=1))
y_pred = model.predict(test)
sub['ADHD_Outcome'] = y_pred[:,0]
sub['Sex_F'] = y_pred[:,1]
sub.to_csv(f'{RESULT_FOLDER}submission.csv',index=False)

#produces pca, umap, piechart pngs