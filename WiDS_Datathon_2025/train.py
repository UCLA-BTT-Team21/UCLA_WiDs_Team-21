import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import umap
import gc
import time
from preprocessing import prepare_data_for_modeling, CACHE_FOLDER, FIGURE_FOLDER, RESULT_FOLDER

# Set random seed for reproducibility
RANDOM_SEED = 42

def apply_smote_to_multi_output(X_train, y_train, random_state=RANDOM_SEED):
    """
    Apply SMOTE to multi-output classification data
    
    Parameters:
    X_train: Training features
    y_train: Multi-output training targets
    random_state: Random state for reproducibility
    
    Returns:
    tuple: X_resampled, y_resampled as DataFrames
    """
    print("Applying SMOTE for multi-output classification...")
    
    # Create a combined target
    # This creates a unique class for each combination of target values
    y_combined = y_train.iloc[:, 0].astype(str)
    for i in range(1, len(y_train.columns)):
        y_combined += '_' + y_train.iloc[:, i].astype(str)
    
    print(f"Original class distribution: {pd.Series(y_combined).value_counts()}")
    
    # Apply SMOTE on the combined target
    smote = SMOTE(random_state=random_state)
    X_res, y_combined_res = smote.fit_resample(X_train, y_combined)
    
    print(f"Resampled class distribution: {pd.Series(y_combined_res).value_counts()}")
    
    # Split the combined target back into individual targets
    y_res = pd.DataFrame(index=range(len(y_combined_res)))
    
    for i, col in enumerate(y_train.columns):
        # Extract the ith position from each combined class
        y_res[col] = y_combined_res.str.split('_').str[i].astype(int)
    
    return X_res, y_res

def train_models(X_train, y_train, X_test, y_test, use_smote=True, n_jobs=-1):
    start_time = time.time()
    
    # Define model configurations with optimized parameters
    model_configs = {
        'LGBM': lgb.LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            n_estimators=200,
            random_state=RANDOM_SEED,
            verbose=-1,
            n_jobs=n_jobs
        ),
        'XGB': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=200,
            random_state=RANDOM_SEED,
            n_jobs=n_jobs,
            tree_method='hist'  # Using histogram-based algorithm for better performance
        ),
        'CatBoost': cb.CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            random_state=RANDOM_SEED,
            silent=True,
            thread_count=n_jobs
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_SEED
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=n_jobs
        )
    }
    
    # Apply SMOTE if requested
    if use_smote:
        print("Preprocessing with SMOTE...")
        X_train_res, y_train_res = apply_smote_to_multi_output(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    
    results = {}
    models = {}
    
    # Train each model and evaluate
    for name, model in model_configs.items():
        print(f"\nTraining {name}...")
        model_start_time = time.time()
        
        # Use MultiOutputClassifier directly
        multi_model = MultiOutputClassifier(model)
        multi_model.fit(X_train_res, y_train_res)
        models[name] = multi_model
        y_pred = multi_model.predict(X_test)
        
        # Evaluate
        f1 = f1_score(y_test, y_pred, average='micro')
        training_time = time.time() - model_start_time
        
        # Store results
        results[name] = {
            'f1_score': f1,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"{name} - F1 Score: {f1:.4f}, Training Time: {training_time:.2f} seconds")
        
        # Generate detailed reports
        for i, target in enumerate(y_train.columns):
            print(f"\n{name} - {target} Report:")
            print(classification_report(y_test[target], y_pred[:, i]))
        
        # Free memory
        gc.collect()
    
    # Create an ensemble model using the best performers
    print("\nTraining Ensemble Model...")
    model_start_time = time.time()
    
    # Select top 3 models based on F1 score
    top_models = sorted([(name, config) for name, config in model_configs.items()], 
                        key=lambda x: results[x[0]]['f1_score'], 
                        reverse=True)[:3]
    
    # Create a voting classifier
    estimators = [(name, model_configs[name]) for name, _ in top_models]
    
    # Use MultiOutputClassifier with VotingClassifier
    ensemble = MultiOutputClassifier(VotingClassifier(estimators=estimators, voting='soft'))
    ensemble.fit(X_train_res, y_train_res)
    models['Ensemble'] = ensemble
    y_pred = ensemble.predict(X_test)
    
    # Evaluate
    f1 = f1_score(y_test, y_pred, average='micro')
    training_time = time.time() - model_start_time
    
    # Store results
    results['Ensemble'] = {
        'f1_score': f1,
        'training_time': training_time,
        'predictions': y_pred
    }
    
    print(f"Ensemble - F1 Score: {f1:.4f}, Training Time: {training_time:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f} seconds")
    
    # Save models and results
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    joblib.dump(models, os.path.join(CACHE_FOLDER, 'trained_models.pkl'))
    joblib.dump(results, os.path.join(CACHE_FOLDER, 'model_results.pkl'))
    
    return models, results

def visualize_results(results):
    """
    Visualize model performance
    
    Parameters:
    results: Dictionary of model results
    """
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    
    # Plot F1 scores
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    f1_scores = [results[model]['f1_score'] for model in models]
    
    ax = sns.barplot(x=models, y=f1_scores)
    ax.set_xlabel('Model')
    ax.set_ylabel('F1 Score')
    ax.set_title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add the values on top of the bars
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, 'model_performance.png'), dpi=300)
    
    # Plot training time
    plt.figure(figsize=(10, 6))
    training_times = [results[model]['training_time'] for model in models]
    
    ax = sns.barplot(x=models, y=training_times)
    ax.set_xlabel('Model')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Model Training Time Comparison')
    plt.xticks(rotation=45)
    
    # Add the values on top of the bars
    for i, v in enumerate(training_times):
        ax.text(i, v + 1, f"{v:.1f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_FOLDER, 'training_time.png'), dpi=300)
    
    # Clear the plots
    plt.close('all')

def generate_submission(model, test_data, submission_template_path=None):
    """
    Generate a submission file
    
    Parameters:
    model: Trained model
    test_data: Test data
    submission_template_path: Path to the submission template file
    """
    # Load preprocessor
    preprocessor = joblib.load(os.path.join(CACHE_FOLDER, 'preprocessor.pkl'))
    
    # Preprocess test data
    test_processed = preprocessor.transform(test_data)
    
    # Make predictions
    predictions = model.predict(test_processed)
    
    # Create submission dataframe
    if submission_template_path:
        submission = pd.read_excel(submission_template_path)
    else:
        submission = pd.DataFrame({'participant_id': test_data.index})
    
    # Add predictions to submission
    submission['ADHD_Outcome'] = predictions[:, 0]
    submission['Sex_F'] = predictions[:, 1]
    
    # Ensure predictions are binary
    submission['ADHD_Outcome'] = submission['ADHD_Outcome'].astype(int)
    submission['Sex_F'] = submission['Sex_F'].astype(int)
    
    # Save submission file
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    submission_path = os.path.join(RESULT_FOLDER, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission saved to {submission_path}")
    
    return submission

def visualize_feature_importance(models, feature_names, top_n=20):
    """
    Visualize feature importance for tree-based models
    
    Parameters:
    models: Dictionary of trained models
    feature_names: List of feature names
    top_n: Number of top features to display
    """
    os.makedirs(FIGURE_FOLDER, exist_ok=True)
    
    for name, model in models.items():
        # Skip ensemble model
        if name == 'Ensemble':
            continue
        
        # Check if the model has SMOTE
        if hasattr(model, 'steps'):
            # Extract the model from the pipeline
            estimator = model.steps[-1][1]
        else:
            estimator = model
        
        try:
            # For tree-based models, extract feature importance
            if name in ['LGBM', 'XGB', 'CatBoost', 'RandomForest', 'GradientBoosting']:
                # Get feature importance from the first estimator (for multi-output)
                if hasattr(estimator, 'estimators_'):
                    importance = estimator.estimators_[0].feature_importances_
                else:
                    importance = estimator.feature_importances_
                
                # Sort feature importance
                indices = np.argsort(importance)[::-1][:top_n]
                
                # Plot
                plt.figure(figsize=(12, 8))
                plt.title(f'Top {top_n} Feature Importance - {name}')
                plt.barh(range(top_n), importance[indices])
                plt.yticks(range(top_n), [feature_names[i] for i in indices])
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURE_FOLDER, f'feature_importance_{name}.png'), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Could not extract feature importance for {name}: {str(e)}")

def main():
    """
    Main function to run the entire pipeline
    """
    print("Starting the pipeline...")
    
    # Prepare data
    print("Preparing data...")
    X_train, X_test, y_train, y_test, test_data = prepare_data_for_modeling(n_features=500)
    
    # Train models
    print("Training models...")
    models, results = train_models(X_train, y_train, X_test, y_test, use_smote=True)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(results)
    
    # Get the best model
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
    best_model = models[best_model_name]
    print(f"Best model: {best_model_name} with F1 score: {results[best_model_name]['f1_score']:.4f}")
    
    # Generate submission
    print("Generating submission...")
    submission = generate_submission(
        best_model, 
        test_data, 
        submission_template_path=os.path.join(DATA_FOLDER, 'SAMPLE_SUBMISSION.xlsx')
    )
    
    # Visualize feature importance
    print("Visualizing feature importance...")
    visualize_feature_importance(models, X_train.columns)
    
    print("Pipeline completed successfully!")
    return models, results, submission

if __name__ == "__main__":
    main()