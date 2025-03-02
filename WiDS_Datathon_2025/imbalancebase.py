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
from sklearn.base import clone  # Add this import
import umap
import gc
import time

# Define folder paths
DATA_FOLDER = 'data/'
CACHE_FOLDER = 'cache/'
FIGURE_FOLDER = 'img/'
RESULT_FOLDER = 'results/'

# Create necessary directories
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Import preprocessing after folder definition to ensure consistency
try:
    from preprocessing import prepare_data_for_modeling
except ImportError:
    # If preprocessing module is not available, use function from this file
    print("preprocessing module not found, using local functions")
    # You would need to define prepare_data_for_modeling here or use an alternative

# Set random seed for reproducibility
RANDOM_SEED = 42

# Custom model classes for multi-output classification
class CustomMultiOutputModel:
    """
    Custom model that handles multiple targets with separate estimators
    """
    def __init__(self, estimators):
        self.estimators = estimators
    
    def predict(self, X):
        return np.column_stack([est.predict(X) for est in self.estimators])
    
    def fit(self, X, y):
        # This is a dummy method since models are already fitted
        return self

class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple base models
    """
    def __init__(self, base_models, model_names):
        self.base_models = base_models
        self.model_names = model_names
    
    def predict(self, X):
        predictions = []
        
        for i in range(len(self.base_models[self.model_names[0]].estimators)):
            target_preds = []
            
            for model_name in self.model_names:
                target_model = self.base_models[model_name].estimators[i]
                target_preds.append(target_model.predict(X))
            
            # Majority vote for binary classification
            ensemble_pred = np.round(np.mean(target_preds, axis=0)).astype(int)
            predictions.append(ensemble_pred)
        
        return np.column_stack(predictions)

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
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=n_jobs
        )
    }
    
    results = {}
    models = {}
    
    # Convert y_train DataFrame into numpy array if it isn't already
    if isinstance(y_train, pd.DataFrame):
        y_train_array = y_train.values
    else:
        y_train_array = y_train
        
    # Convert y_test DataFrame into numpy array if it isn't already
    if isinstance(y_test, pd.DataFrame):
        y_test_array = y_test.values
    else:
        y_test_array = y_test
    
    # Train each model and evaluate
    for name, model in model_configs.items():
        print(f"\nTraining {name}...")
        model_start_time = time.time()
        
        # Implement a simpler approach - train separate models for each target
        # This avoids SMOTE multi-output issues
        estimators = []
        predictions = []
        
        for i, target_name in enumerate(y_train.columns if isinstance(y_train, pd.DataFrame) else range(y_train_array.shape[1])):
            print(f"Training for target: {target_name}")
            
            # Get the target column
            if isinstance(y_train, pd.DataFrame):
                y_target = y_train[target_name]
                y_test_target = y_test[target_name]
            else:
                y_target = y_train_array[:, i]
                y_test_target = y_test_array[:, i]
            
            # Create a clone of the model for this target
            target_model = clone(model)
            
            if use_smote:
                # Apply SMOTE for this specific target
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=RANDOM_SEED)
                X_res, y_res = smote.fit_resample(X_train, y_target)
                
                # Train on resampled data
                target_model.fit(X_res, y_res)
            else:
                # Train directly
                target_model.fit(X_train, y_target)
            
            # Store the trained model
            estimators.append(target_model)
            
            # Predict
            y_pred_target = target_model.predict(X_test)
            predictions.append(y_pred_target)
            
            # Evaluate individual performance
            target_f1 = f1_score(y_test_target, y_pred_target, average='micro')
            print(f"{name} - {target_name} F1 Score: {target_f1:.4f}")
        
        # Combine predictions
        y_pred = np.column_stack(predictions)
        
        # Create a custom multi-output model (not using sklearn's MultiOutputClassifier)
        models[name] = CustomMultiOutputModel(estimators)
        
        # Evaluate overall
        f1 = f1_score(y_test_array, y_pred, average='micro')
        training_time = time.time() - model_start_time
        
        # Store results
        results[name] = {
            'f1_score': f1,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"{name} - Overall F1 Score: {f1:.4f}, Training Time: {training_time:.2f} seconds")
        
        # Generate detailed reports
        for i, target_name in enumerate(y_train.columns if isinstance(y_train, pd.DataFrame) else range(y_train_array.shape[1])):
            print(f"\n{name} - {target_name} Report:")
            if isinstance(y_test, pd.DataFrame):
                target_test = y_test[target_name]
            else:
                target_test = y_test_array[:, i]
            print(classification_report(target_test, y_pred[:, i]))
        
        # Free memory
        gc.collect()
    
    # Create an ensemble model using the best performers
    print("\nTraining Ensemble Model...")
    model_start_time = time.time()
    
    # Select top 3 models based on F1 score
    top_models_names = sorted(results.keys(), key=lambda x: results[x]['f1_score'], reverse=True)[:3]
    
    # Create ensemble predictions by averaging the predictions of top models
    ensemble_predictions = []
    
    for i in range(y_train_array.shape[1]):
        # For each target, create an ensemble of predictions
        target_ensemble_preds = []
        
        for model_name in top_models_names:
            target_ensemble_preds.append(results[model_name]['predictions'][:, i])
        
        # Average the predictions (for binary classification we use majority vote)
        ensemble_pred = np.round(np.mean(target_ensemble_preds, axis=0)).astype(int)
        ensemble_predictions.append(ensemble_pred)
    
    # Stack predictions
    y_pred_ensemble = np.column_stack(ensemble_predictions)
    
    # Create the ensemble model
    ensemble_model = EnsembleModel(models, top_models_names)
    models['Ensemble'] = ensemble_model
    
    # Evaluate
    f1 = f1_score(y_test_array, y_pred_ensemble, average='micro')
    training_time = time.time() - model_start_time
    
    # Store results
    results['Ensemble'] = {
        'f1_score': f1,
        'training_time': training_time,
        'predictions': y_pred_ensemble
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
        # Skip ensemble model or models without feature importance
        if name == 'Ensemble':
            continue
        
        try:
            # For tree-based models, extract feature importance
            if name in ['LGBM', 'XGB', 'CatBoost', 'RandomForest']:
                # Get the first estimator (assuming it's for the first target)
                if hasattr(model, 'estimators'):
                    # For our custom multi-output model
                    base_model = model.estimators[0]
                elif hasattr(model, 'estimators_'):
                    # For sklearn's MultiOutputClassifier
                    base_model = model.estimators_[0]
                else:
                    print(f"Cannot extract feature importance for {name}: unknown model structure")
                    continue
                
                # Different models store feature importance differently
                if hasattr(base_model, 'feature_importances_'):
                    importance = base_model.feature_importances_
                elif name == 'LGBM' and hasattr(base_model, 'booster_'):
                    # LightGBM specific
                    importance = base_model.booster_.feature_importance()
                elif name == 'XGB' and hasattr(base_model, 'get_booster'):
                    # XGBoost specific
                    importance = base_model.get_booster().get_score(importance_type='gain')
                    # Convert to array format
                    if isinstance(importance, dict):
                        # Convert dict to array, filling with zeros for missing features
                        temp_importance = np.zeros(len(feature_names))
                        for feat, imp in importance.items():
                            try:
                                feat_idx = int(feat.replace('f', ''))
                                if feat_idx < len(temp_importance):
                                    temp_importance[feat_idx] = imp
                            except:
                                pass
                        importance = temp_importance
                else:
                    print(f"Cannot extract feature importance for {name}: no feature_importances_ attribute")
                    continue
                
                # Ensure we don't have dimension mismatch
                if len(importance) > len(feature_names):
                    importance = importance[:len(feature_names)]
                elif len(importance) < len(feature_names):
                    # Pad with zeros if needed
                    padded_importance = np.zeros(len(feature_names))
                    padded_importance[:len(importance)] = importance
                    importance = padded_importance
                
                # Sort feature importance
                indices = np.argsort(importance)[::-1][:min(top_n, len(importance))]
                
                # Plot
                plt.figure(figsize=(12, 8))
                plt.title(f'Top {min(top_n, len(indices))} Feature Importance - {name}')
                plt.barh(range(len(indices)), importance[indices])
                plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in indices])
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(os.path.join(FIGURE_FOLDER, f'feature_importance_{name}.png'), dpi=300)
                plt.close()
                
                print(f"Feature importance visualization created for {name}")
        except Exception as e:
            print(f"Could not extract feature importance for {name}: {str(e)}")
            import traceback
            traceback.print_exc()

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
    
    # # Visualize feature importance
    # print("Visualizing feature importance...")
    # visualize_feature_importance(models, X_train.columns)
    
    print("Pipeline completed successfully!")
    return models, results, submission

if __name__ == "__main__":
    main()