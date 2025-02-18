import numpy as np 
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import scipy
from utils import get_feats, check_for_nulls

#Constants
DATA_FOLDER = 'data/'
FIGURE_FOLDER = 'img/'
RESULT_FOLDER = 'results/'
RANDOM_STATE = 42

# Load data
train = get_feats(mode='TRAIN')
test = get_feats(mode='TEST')
sub = pd.read_excel('data/SAMPLE_SUBMISSION.xlsx')
y = pd.read_excel(f"data/TRAIN/TRAINING_SOLUTIONS.xlsx")

# Data preprocessing
train.set_index('participant_id', inplace=True)
test.set_index('participant_id', inplace=True)
targets = ['ADHD_Outcome', 'Sex_F']
features = test.columns

# Identify features for log transformation
log_features = [f for f in features if (train[f] >= 0).all() and scipy.stats.skew(train[f]) > 0]

# preprocessing pipeline
preprocessor = make_pipeline(
    ColumnTransformer(
        [('imputer', SimpleImputer(strategy='median'), features)],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform='pandas'),
    ColumnTransformer(
        [('log', FunctionTransformer(np.log1p), log_features)],
        remainder='passthrough'
    )
)

# Prepare the data
X = train.drop(targets, axis=1)
y_full = y[targets]

# LightGBM parameters
lgb_params_adhd = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'random_state': RANDOM_STATE,
    'n_estimators': 200,
    'verbose': -1
}

lgb_params_sex = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'random_state': RANDOM_STATE,
    'n_estimators': 200,
    'verbose': -1
}

# Calculate class weights for ADHD (extra weight for female ADHD cases)
def get_class_weights(y_adhd, y_sex):
    weights = np.ones(len(y_adhd))
    # Give 2x weight to female ADHD cases
    weights[(y_adhd == 1) & (y_sex == 1)] = 2
    return weights

# Prepare cross-validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

# Lists to store predictions
oof_preds_adhd = np.zeros(len(X))
oof_preds_sex = np.zeros(len(X))
test_preds_adhd = np.zeros(len(test))
test_preds_sex = np.zeros(len(test))

# Train and predict with cross-validation
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_full['ADHD_Outcome'])):
    print(f'Training fold {fold + 1}/{n_splits}')
    
    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train_adhd, y_val_adhd = y_full['ADHD_Outcome'].iloc[train_idx], y_full['ADHD_Outcome'].iloc[val_idx]
    y_train_sex, y_val_sex = y_full['Sex_F'].iloc[train_idx], y_full['Sex_F'].iloc[val_idx]
    
    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(test)
    
    # Calculate sample weights
    sample_weights = get_class_weights(y_train_adhd, y_train_sex)
    
    # Train ADHD model
    model_adhd = lgb.LGBMClassifier(**lgb_params_adhd)
    model_adhd.fit(
        X_train_processed, y_train_adhd,
        sample_weight=sample_weights,
        eval_set=[(X_val_processed, y_val_adhd)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Train Sex model
    model_sex = lgb.LGBMClassifier(**lgb_params_sex)
    model_sex.fit(
        X_train_processed, y_train_sex,
        eval_set=[(X_val_processed, y_val_sex)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Store predictions
    oof_preds_adhd[val_idx] = model_adhd.predict_proba(X_val_processed)[:, 1]
    oof_preds_sex[val_idx] = model_sex.predict_proba(X_val_processed)[:, 1]
    
    test_preds_adhd += model_adhd.predict_proba(X_test_processed)[:, 1] / n_splits
    test_preds_sex += model_sex.predict_proba(X_test_processed)[:, 1] / n_splits

# Convert probabilities to binary predictions
sub['ADHD_Outcome'] = (test_preds_adhd > 0.5).astype(int)
sub['Sex_F'] = (test_preds_sex > 0.5).astype(int)

# Save predictions
sub.to_csv(f'{RESULT_FOLDER}submission.csv', index=False)

# Calculate and print validation scores
print('Validation F1 Scores:')
print(f'ADHD: {f1_score(y_full["ADHD_Outcome"], (oof_preds_adhd > 0.5).astype(int))}')
print(f'Sex: {f1_score(y_full["Sex_F"], (oof_preds_sex > 0.5).astype(int))}')

# Feature importance plots
plt.figure(figsize=(10, 6))
lgb.plot_importance(model_adhd, max_num_features=20)
plt.title('ADHD Model Feature Importance')
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}adhd_feature_importance.png", dpi=300)
plt.clf()

plt.figure(figsize=(10, 6))
lgb.plot_importance(model_sex, max_num_features=20)
plt.title('Sex Model Feature Importance')
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}sex_feature_importance.png", dpi=300)
plt.clf()

#note: made sex and adhd feature importance graphics