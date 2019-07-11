import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import pickle
import os
import gc
import sys
gc.enable()

def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    
    model = xgb.XGBClassifier(max_depth=2,
                              n_estimators=999999,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='binary:logistic', 
                              n_jobs=-1)
     
    model.fit(X_fit, y_fit, 
              eval_set=[(X_val, y_val)], 
              verbose=0, 
              early_stopping_rounds=1000)
              
    cv_val = model.predict_proba(X_val)[:,1]
    
    #Save XGBoost Model
    save_to = '{}{}_fold{}.dat'.format(xgb_path, name, counter+1)
    pickle.dump(model, open(save_to, "wb"))
    
    return cv_val


def train_stage(df_path, xgb_path, limit_row):
    
    print('Load Train Data.')
    df = pd.read_csv(df_path)[0:limit_row]
    print('\nShape of Train Data: {}'.format(df.shape))
    
    y_df = np.array(df['target'])                        
    df_ids = np.array(df.index)                     
    df.drop(['ID_code', 'target'], axis=1, inplace=True)

    xgb_cv_result = np.zeros(df.shape[0])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]

        print('XGBoost')
        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()

    auc_xgb = round(roc_auc_score(y_df, xgb_cv_result),4)

    print('\nXGBoost  VAL AUC: {}'.format(auc_xgb))
    
    return 0
    
    
def prediction_stage(df_path, xgb_path, limit_row):
    
    print('Load Test Data.')
    df = pd.read_csv(df_path)
    print('\nShape of Test Data: {}'.format(df.shape))
    
    df.drop(['ID_code'], axis=1, inplace=True)

    xgb_models = sorted(os.listdir(xgb_path))

    xgb_result = np.zeros(df.shape[0])
    
    print('\nMake predictions...\n')

    print('With XGBoost...')    
    for m_name in xgb_models:
        #Load XGBoost Model
        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
        xgb_result += model.predict_proba(df.values)[:,1]

    xgb_result /= len(xgb_models)

    # submission = pd.read_csv('../input/sample_submission.csv')
    # submission.to_csv('lgb_cb_starter_submission.csv', index=False)
    
    return 0
    
    
if __name__ == '__main__':

    num_params = len(sys.argv)
    limit_row = None
    if num_params < 2:
        print("No arguments are passed. Using default.")
    else:
        if num_params > 1:
            limit_row = int(sys.argv[1])
            print("Reducing times by choosing first {0} elements.".format(limit_row))

    train_path = '../data/train.csv'
    test_path  = '../data/test.csv'
    xgb_path = './Models/'

    print('Train Stage.\n')
    train_stage(train_path, xgb_path, limit_row)
    
    print('Prediction Stage.\n')
    prediction_stage(test_path, xgb_path, limit_row)
    
    print('\nDone.')