import pandas as pd

def get_feats(mode='train'):
   
    feats=pd.read_excel(f"data/{mode}/{mode}_QUANTITATIVE_METADATA.xlsx")
    if mode=='TRAIN':
        cate=pd.read_excel(f"data/{mode}/{mode}_CATEGORICAL_METADATA.xlsx")
    elif mode == 'TEST':
        cate=pd.read_excel(f"data/{mode}/{mode}_CATEGORICAL.xlsx")
    else:
        raise NotImplementedError
    
    feats=feats.merge(cate,on='participant_id',how='left')
    
    func=pd.read_csv(f"data/{mode}/{mode}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    feats=feats.merge(func,on='participant_id',how='left')

    if mode=='TRAIN':
        solution=pd.read_excel("data/TRAIN/TRAINING_SOLUTIONS.xlsx")
        feats=feats.merge(solution,on='participant_id',how='left')
        
    return feats

def check_for_nulls(df):
  """
  Checks for null values in a pandas DataFrame and prints a message.

  Args:
    df: The pandas DataFrame to check.

  Returns:
    None
  """
  if df.isnull().any().any():
    print("The DataFrame contains null values.")
  else:
    print("The DataFrame does not contain null values.")