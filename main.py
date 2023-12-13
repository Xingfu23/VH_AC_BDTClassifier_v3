import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import xgboost as xgb
import optuna
from optuna import Trial
import argparse

from tools.xgboost2tmva import *
from tools.bdt_vars import *
from tools.common_tool import file_exsit, colloect_samples, output_xmlfile
from plot_tools.plot_type import Fitting_Plots, model_importance_plot, probability_plot, roc_curve


def get_args()-> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--plotfoldername', help="Folder's Name for output plot and xml file", type=str, default='TestName')
    parser.add_argument('-ac', '--ac', help='The type of ac, there are 3 types: fa2, fa3, fL1', type=str)
    parser.add_argument('-x', '--xmlfile', help='Output xml file or not', type=bool, default=False)
    parser.add_argument('-op', '--outplot', help='Output plot or not', type=bool, default=False)

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    
    # Check args
    ac_type_option = ['fa2', 'fa3', 'fL1']
    if args.ac not in ac_type_option:
        print(f"Please choose the type of ac from {ac_type_option}")
        return 0
    
    # Collect background (SM VH) and signal (AC VH) samples and combine each of them into a single dataframe
    select_dataset = dataset
    df_bkg = colloect_samples(0, select_dataset)
    df_sig = colloect_samples(1, select_dataset, args.ac)
    
    # Mark the background and signal variables and caluate the entry number of each
    df_bkg['sig/bkg'] = 0
    df_sig['sig/bkg'] = 1
    postive_ratio = len(df_bkg)/len(df_sig)
    print(f"Background/Signal ratio: {postive_ratio:.4f}\n")
    df_dataset = pd.concat([df_bkg, df_sig], ignore_index=True, axis=0)
    
    X = df_dataset[dataset]
    y = df_dataset['sig/bkg']
    
    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=69)
    data_DMatrix_train = xgb.DMatrix(data=X_train, label=y_train)
    eval_set = [(X_train, y_train)]
    
    # TODO Fine tuning the hyperparameters
    def objective(trial, X_train=X_train, y_train=y_train):
        tree_method_option = ['gpu_hist']
        boosting_type_option = ['dart', 'gbtree']
        objective_option = ['count:poisson', 'binary:logistic']
        metric_option = ['logloss']
        param = {
            'booster': trial.suggest_categorical('booster', boosting_type_option),
            'tree_method': trial.suggest_categorical('tree_method', tree_method_option),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.),
            'gamma': trial.suggest_float('gamma', 1, 10),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 0, 10),
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
            'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1),
            'alpha': trial.suggest_float('reg_alpha', 0, 10),
            'lambda': trial.suggest_float('reg_lambda', 0, 10),
            'objective': trial.suggest_categorical('objective', objective_option),
            'eval_metric': trial.suggest_categorical('eval_metric', metric_option),
        }
        xgb_model = xgb.XGBClassifier(
            **param,
            use_label_encoder = None,
            early_stopping_rounds = 20,
            Scale_pos_weight = postive_ratio,
            gpu_id = 0, # Using GPU
        )
        
        # Training
        xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return metrics.log_loss(y_test, xgb_model.predict_proba(X_test))
    
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=69), direction='minimize')
    study.optimize(objective, n_trials=50)
    
    # Print the study results: boosters, tree_methods, max_depths, learning_rates, values
    df_result = study.trials_dataframe()
    
    trial = study.best_trial
    _best_params = trial.params
    
    # KFold cross validation, and setup the best xgb model 
    kf = KFold(n_splits=5, shuffle=True, random_state=69)
    
    logloss_train = []
    logloss_test = []
    
    xgb_model = xgb.XGBClassifier(
        **_best_params,
        use_label_encoder = None,
        early_stopping_rounds = 20,
        scale_pos_weight = postive_ratio,
        gpu_id = 0, # Using GPU
    )
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        y_pred_train = xgb_model.predict_proba(X_train)
        y_pred_test = xgb_model.predict_proba(X_test)
        logloss_train.append(metrics.log_loss(y_train, y_pred_train))
        logloss_test.append(metrics.log_loss(y_test, y_pred_test))
        
    print(f"Logloss of train set: {np.array(logloss_train).mean():.4f} +- {np.array(logloss_train).std():.4f}")
    
    # Create output folder if not exist
    plotfolder = args.plotfoldername
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
        print(f"Create folder: {plotfolder}")
        
    # Saving the best model as txt file
    with open(f'{plotfolder}/bestmodel_results.txt', 'w') as file:
        file.write(f"Best hyperparameters: {_best_params}\n")
        file.write(f"Best score: {study.best_value}\n")
        print(f"Logloss of train set: {np.array(logloss_train).mean():.4f} +- {np.array(logloss_train).std():.4f}", file=file)
    
    # Visualize the results to check overfitting
    if args.outplot:
        Fitting_Plots(logloss_train, logloss_test, plotfolder)
    
    # Output the xml file
    if args.xmlfile:
        output_xmlfile(plotfolder, xgb_model, dataset_forxml)
    else:
        print("No xml file output.")
    
    # Make predictions
    y_pred = pd.DataFrame(xgb_model.predict(X_test), columns=['sig/bkg'])
    y_pred_prob_train = pd.DataFrame(xgb_model.predict_proba(X_train))
    y_pred_prob_test = pd.DataFrame(xgb_model.predict_proba(X_test))
    
    print(" ")
    print(f'Train group: {xgb_model.score(X_train,y_train):.4f}')
    print(f'Test group: {xgb_model.score(X_test,y_test):.4f}')
    
    # Make Plots
    if args.outplot:
        print("Plotting...")
        
        y_valid_dict = {'Train': [y_train, y_pred_prob_train], 'Test': [y_test, y_pred_prob_test]}
        
        model_importance_plot(xgb_model, plotfolder) # Drawing importance plot
        probability_plot(y_pred_prob_test, y_test, plotfolder, args.ac) # Make probabilities histograms
        roc_curve(y_valid_dict, plotfolder) # Plot ROC curve
        


if __name__ == "__main__":
    '''
    command example: python3 main.py -n "acbdt_fa31d0" -ac "fa3" -x 1 -op 1
    '''
    main()