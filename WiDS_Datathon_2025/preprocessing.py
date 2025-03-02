import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# Define constants
DATA_FOLDER = 'data/'
FIGURE_FOLDER = 'img/'
RESULT_FOLDER = 'results/'
CACHE_FOLDER = 'cache/'

# Create directories if they don't exist
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def get_feats(mode='TRAIN', use_cache=True, reduce_dim=True, n_features=500):
    cache_file = os.path.join(CACHE_FOLDER, f'{mode.lower()}_preprocessed_{n_features}.pkl')
    
    # Return cached data if available and requested
    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached {mode} data...")
        return joblib.load(cache_file)
    
    print(f"Processing {mode} data from scratch...")
    
    # Load connectome matrices - chunking to reduce memory usage
    if mode == 'TRAIN':
        file_path = os.path.join(DATA_FOLDER, 'TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
    else:
        file_path = os.path.join(DATA_FOLDER, 'TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv')
    
    # Use chunking to load the large CSV file - helps with memory issues
    chunks = []
    for chunk in tqdm(pd.read_csv(file_path, chunksize=50), desc=f"Loading {mode} data"):
        chunks.append(chunk)
    
    df = pd.concat(chunks, axis=0)
    print(f"Data loaded: {df.shape}")
    
    # Free memory
    del chunks
    gc.collect()
    
    # Extract subject IDs
    subject_ids = df['participant_id'].copy()
    
    # Find connectome matrix columns (throw columns)
    connectome_cols = [col for col in df.columns if 'throw' in col]
    
    # Handle missing values in connectome matrices
    print("Handling missing values...")
    null_counts = df[connectome_cols].isnull().sum()
    print(f"Columns with nulls: {len(null_counts[null_counts > 0])}")
    
    # Keep only participant_id and connectome columns
    df = df[['participant_id'] + connectome_cols]
    
    # Optimize DataFrame memory usage
    df = optimize_dataframe_memory(df)
    
    # Apply feature selection if requested
    if reduce_dim and mode == 'TRAIN':
        print(f"Reducing dimensions to {n_features} features...")
        # Load target data for feature selection
        y_data = pd.read_excel(os.path.join(DATA_FOLDER, 'TRAIN/TRAINING_SOLUTIONS.xlsx'))
        
        # Merge with current data to align samples
        merged = pd.merge(df, y_data, on='participant_id', how='inner')
        X = merged[connectome_cols]
        y_adhd = merged['ADHD_Outcome']
        
        # Use SelectKBest to choose most discriminative features
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(X, y_adhd)
        
        # Get selected feature names
        selected_features = np.array(connectome_cols)[selector.get_support()]
        print(f"Selected {len(selected_features)} features")
        
        # Save the selector for test data
        joblib.dump(selector, os.path.join(CACHE_FOLDER, 'feature_selector.pkl'))
        
        # Keep only selected features
        df = df[['participant_id'] + selected_features.tolist()]
        
    elif reduce_dim and mode == 'TEST':
        # For test data, use the selector trained on training data
        selector = joblib.load(os.path.join(CACHE_FOLDER, 'feature_selector.pkl'))
        selected_features = np.array(connectome_cols)[selector.get_support()]
        df = df[['participant_id'] + selected_features.tolist()]
    
    print(f"Final shape: {df.shape}")
    
    # Set participant_id as index
    df.set_index('participant_id', inplace=True)
    
    # Cache the preprocessed data
    joblib.dump(df, cache_file)
    print(f"Cached {mode} data to {cache_file}")
    
    return df

def optimize_dataframe_memory(df):
    """
    Optimize the memory usage of a DataFrame by using appropriate dtypes
    """
    # For float columns, convert to float32 instead of float64
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float32')
    
    # For integer columns, determine the appropriate integer type
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        max_val = df[col].max()
        min_val = df[col].min()
        
        if min_val >= 0:
            if max_val < 2**8:
                df[col] = df[col].astype('uint8')
            elif max_val < 2**16:
                df[col] = df[col].astype('uint16')
            elif max_val < 2**32:
                df[col] = df[col].astype('uint32')
        else:
            if min_val > -2**7 and max_val < 2**7:
                df[col] = df[col].astype('int8')
            elif min_val > -2**15 and max_val < 2**15:
                df[col] = df[col].astype('int16')
            elif min_val > -2**31 and max_val < 2**31:
                df[col] = df[col].astype('int32')
    
    return df

def check_for_nulls(df):
    """Check for null values in the DataFrame and return statistics"""
    null_values = df.isnull().sum().sum()
    if null_values > 0:
        print(f"The DataFrame contains {null_values} null values.")
        print("Columns with null values:")
        print(df.columns[df.isnull().any()].tolist())
        return True
    else:
        print("No null values found.")
        return False

def prepare_data_for_modeling(n_features=500, test_size=0.3, random_state=42):
    """
    Prepare data for modeling, including preprocessing and train-test split
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, test_data
    """
    # Load and preprocess data
    train_data = get_feats(mode='TRAIN', reduce_dim=True, n_features=n_features)
    test_data = get_feats(mode='TEST', reduce_dim=True, n_features=n_features)
    
    # Load target data
    y_data = pd.read_excel(os.path.join(DATA_FOLDER, 'TRAIN/TRAINING_SOLUTIONS.xlsx'))
    y_data.set_index('participant_id', inplace=True)
    
    # Make sure indices match
    common_indices = train_data.index.intersection(y_data.index)
    train_data = train_data.loc[common_indices]
    y_data = y_data.loc[common_indices]
    
    # Split the data
    targets = ['ADHD_Outcome', 'Sex_F']
    X_train, X_test, y_train, y_test = train_test_split(
        train_data, y_data[targets], 
        test_size=test_size, 
        random_state=random_state,
        stratify=y_data[targets]
    )
    
    # Check for null values
    check_for_nulls(X_train)
    check_for_nulls(X_test)
    check_for_nulls(test_data)
    
    # Identify skewed features for log transformation
    log_features = [f for f in X_train.columns if (X_train[f] >= 0).all() and np.abs(scipy.stats.skew(X_train[f])) > 1]
    print(f"Identified {len(log_features)} skewed features for log transformation")
    
    # Create preprocessor
    preprocessor = make_pipeline(
        ColumnTransformer([
            ('imputer', SimpleImputer(strategy='median'), X_train.columns)
        ], remainder='passthrough'),
        ColumnTransformer([
            ('log', FunctionTransformer(np.log1p), log_features)
        ], remainder='passthrough'),
        StandardScaler()
    )
    
    # Fit and transform training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor for later use
    joblib.dump(preprocessor, os.path.join(CACHE_FOLDER, 'preprocessor.pkl'))
    
    print("Data preparation complete:")
    print(f"X_train shape: {X_train_processed.shape}")
    print(f"X_test shape: {X_test_processed.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, test_data

def apply_smote(X_train, y_train, random_state=42, sampling_strategy='auto'):
    """
    Apply SMOTE to handle class imbalance
    
    Parameters:
    X_train: Features
    y_train: Target variables
    
    Returns:
    tuple: X_resampled, y_resampled
    """
    # For multi-output classification, apply SMOTE separately for each target
    targets = y_train.columns
    X_res = X_train.copy()
    y_res = {}
    
    for target in targets:
        print(f"Applying SMOTE for {target}...")
        smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)
        X_target_res, y_target_res = smote.fit_resample(X_res, y_train[target])
        X_res = X_target_res
        y_res[target] = y_target_res
    
    # Combine the results
    y_res_df = pd.DataFrame(y_res)
    
    return X_res, y_res_df

if __name__ == "__main__":
    # Example usage
    X_train, X_test, y_train, y_test, test_data = prepare_data_for_modeling(n_features=500)
    
    # Apply SMOTE for class imbalance
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    
    print("Data ready for modeling!")
    print(f"Original class distribution: {y_train['ADHD_Outcome'].value_counts()}")
    print(f"Resampled class distribution: {y_train_res['ADHD_Outcome'].value_counts()}")