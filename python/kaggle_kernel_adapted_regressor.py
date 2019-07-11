import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import os
import gc
import sys
gc.enable()

def fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name):
    if not only_regressor:
        classifier = xgb.XGBClassifier(max_depth=2,
                              n_estimators=999999,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='binary:logistic',
                              n_jobs=-1)

    regressor = xgb.XGBRegressor(max_depth=2,
                              n_estimators=999999,
                              colsample_bytree=0.3,
                              learning_rate=0.02,
                              objective='binary:logistic', 
                              n_jobs=-1)
    if not only_regressor:
        classifier.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)

    regressor.fit(X_fit, y_fit,
              eval_set=[(X_val, y_val)],
              verbose=0,
              early_stopping_rounds=1000)


    # Evaluate models
    if not only_regressor:
        classifier_name = 'xgb_fold_classifier_{}'.format(counter+1)
        mean_error_classifier = np.array(classifier.evals_result()['validation_0']['error']).mean()
        print("Evaluation: mean error for model {0}: {1}".format(classifier_name, mean_error_classifier))


    regressor_name = 'xgb_fold_regressor_{}'.format(counter+1)
    mean_error_regressor = np.array(regressor.evals_result()['validation_0']['error']).mean()
    print("Evaluation: mean error for model {0}: {1}".format(regressor_name, mean_error_regressor))

    #Save XGBoost Model
    if not only_regressor:
        save_to_classifier = '{}{}.dat'.format(xgb_path, classifier_name)
        pickle.dump(classifier, open(save_to_classifier, "wb"))

    save_to_regressor = '{}{}.dat'.format(xgb_path, regressor_name)
    pickle.dump(regressor, open(save_to_regressor, "wb"))

    # Return probabilities
    if not only_regressor:
        cv_val = classifier.predict_proba(X_val)[:,1]
        return cv_val

    return 0

def train_stage(df_train, xgb_path, limit_row):
    
    print('Load Train Data.')
    df = df_train
    if limit_row != None:
        df = df_train[0:limit_row].copy()
    print('\nShape of Train Data: {}'.format(df.shape))

    y_df = np.array(df['target'])
    df_ids = np.array(df.index)
    df.drop(['ID_code', 'target'], axis=1, inplace=True)

    xgb_cv_result = np.zeros(df.shape[0])

    skf = StratifiedKFold(n_splits=num_models, shuffle=True, random_state=42)
    skf.get_n_splits(df_ids, y_df)
    
    print('\nModel Fitting...')
    for counter, ids in enumerate(skf.split(df_ids, y_df)):
        print('\nFold {}'.format(counter+1))
        X_fit, y_fit = df.values[ids[0]], y_df[ids[0]]
        X_val, y_val = df.values[ids[1]], y_df[ids[1]]

        xgb_cv_result[ids[1]] += fit_xgb(X_fit, y_fit, X_val, y_val, counter, xgb_path, name='xgb')
        
        del X_fit, X_val, y_fit, y_val
        gc.collect()

    if not only_regressor:
        auc_xgb = round(roc_auc_score(y_df, xgb_cv_result),4)
        print('\nXGBoost VAL AUC: {}'.format(auc_xgb))
    
    return 0


def prediction_stage(df_test, xgb_path, limit_row):
    print('Load Test Data.')
    df = df_test
    if limit_row != None:
        df = df_test[0:limit_row].copy()

    print('\nShape of Test Data: {}'.format(df.shape))
    df.drop(['ID_code', 'target'], axis=1, inplace=True)

    print('\nMake predictions...\n')

    ## Classifiers
    if not only_regressor:
        print('Predict for classifiers')
        xgb_result_proba_classifiers = np.zeros(df.shape[0])
        xgb_result_predict_classifiers = np.zeros(df.shape[0])
        xgb_classifiers = ["xgb_fold_classifier_{}.dat".format(i) for i in range(1,num_models+1)]

        for m_name in xgb_classifiers:
            print("Open {}{} model".format(xgb_path, m_name))
            model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
            xgb_result_proba_classifiers += model.predict_proba(df.values)[:, 1]
            xgb_result_predict_classifiers += model.predict(df.values)

        xgb_result_proba_classifiers /= len(xgb_classifiers)
        xgb_result_predict_classifiers = (xgb_result_predict_classifiers / len(xgb_classifiers)).round(0)

        print("xgb_result_proba_classifiers")
        print(xgb_result_proba_classifiers)
        print("xgb_result_predict_classifiers")
        print(xgb_result_predict_classifiers)

    ## Regressors
    print('Predict for regressors')
    xgb_result_predict_regressors = np.zeros(df.shape[0])
    xgb_regressors = ["xgb_fold_regressor_{}.dat".format(i) for i in range(1,num_models+1)]

    for m_name in xgb_regressors:
        model = pickle.load(open('{}{}'.format(xgb_path, m_name), "rb"))
        prediction = model.predict(df.values)
        print("Predictions for {0}".format(m_name))
        print(prediction)
        print("\n")
        xgb_result_predict_regressors += prediction

    print("Average")
    print(xgb_result_predict_regressors / len(xgb_regressors))

    xgb_result_predict_regressors = (xgb_result_predict_regressors / len(xgb_regressors)).round(0)

    print("xgb_result_predict_regressors")
    print(xgb_result_predict_regressors)

    ## Compare
    if not only_regressor:
        print("Compare results from classifiers and regressors")
        diff = abs(xgb_result_predict_classifiers - xgb_result_predict_regressors)
        mean = diff.mean()
        print("Mean difference between classifier and regressor: {0}".format(mean))

    return 0
    
    
if __name__ == '__main__':

    num_params = len(sys.argv)
    limit_row = None
    only_regressor = True
    num_models = 3
    if num_params < 2:
        print("No arguments are passed. Using default.")
    else:
        if num_params > 1:
            if sys.argv[1] != "all":
                limit_row = int(sys.argv[1])
                print("Reducing times by choosing first {0} elements.".format(limit_row))
            else:
                print("Using full data set")
        if num_params > 2: 
            num_models = int(sys.argv[2])
            print("Creating {0} models for each category (classifier/regressor)".format(num_models))
        if num_params > 3:
            if sys.argv[3] == "only_regressor":
                only_regressor = True
                print("Creating only regressor (no classifier)")


    input_path = '../data/train.csv'
    input = pd.read_csv(input_path)
    # No need to split between train and test, we select evaluation set during training
    #train, test = train_test_split(input, test_size=0.2)
    train = input
    test = pd.read_csv("../data/test.csv")

    xgb_path = './Models/'

    #print('Train Stage.\n')
    #train_stage(train, xgb_path, limit_row)

    print('\nPrediction Stage.\n')
    prediction_stage(train, xgb_path, limit_row)
    
    print('\nDone.')