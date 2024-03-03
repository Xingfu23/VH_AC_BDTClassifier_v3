import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import KFold
import mplhep as hep

def Setup_Plot_Style()->None:
    lumi_runII = 137.19
    fig, ax = plt.subplots(figsize=(8, 6))
    hep.style.use("CMS")
    hep.cms.label(loc=2, data=False, lumi=lumi_runII, fontsize=16)
    
    return fig, ax

def model_importance_plot(model, plotfoldername:str)->None:
    fig, ax = plt.subplots(figsize=(16, 10))
    featvar_im = model.feature_importances_
    xgb.plot_importance(model, ax = ax, height = 0.8, importance_type = 'weight', show_values=False)
    ax.set_xlabel("F score", fontsize=20, fontweight ='bold', loc='right')
    outputname = [f"{plotfoldername}/importance.png", f"{plotfoldername}/importance.pdf"]
    plt.savefig(outputname[0], bbox_inches='tight')
    plt.savefig(outputname[1], bbox_inches='tight')
    print(f"Plotting the importance plot: {outputname[0]}")
    plt.close()

def probability_plot(y_pred_prob_test, y_test, plotfoldername:str, ac_type:str)->None:
    ax, fig = Setup_Plot_Style()
    ax.gca()
    y_pred_com = pd.concat([y_pred_prob_test, y_test.reset_index(drop=True)], axis=1).dropna()
    df_histsig = y_pred_com[y_pred_com['sig/bkg'] == 1]
    df_histbkg = y_pred_com[y_pred_com['sig/bkg'] == 0]
    if ac_type == 'fa3':
        labels = [r"$f_{a1}=1.0$ SM CP-even", r"$f_{a3}=1.0$ CP-odd"]
    elif ac_type == 'fa2':
        labels = [r"$f_{a1}=1.0$ SM", r"$f_{a2}=1.0$"]
    elif ac_type == 'fL1':
        labels = [r"$f_{a1}=1.0$ SM", r"$f_{\Lambda1}=1.0$"]
    else:
        raise ValueError(f"ac_type should be one of {['fa2', 'fa3', 'fL1']}")
    
    bins = np.linspace(0., 1., 50)
    plt.hist(df_histbkg[1], bins, density=True, alpha=0.7, color='b', label=labels[0], log=False)
    plt.hist(df_histsig[1], bins, density=True, alpha=0.7, color='r', label=labels[1], log=False)
    
    plt.xlabel("Probability", fontsize=16, fontweight ='bold', loc='right')
    plt.ylabel("1/Events", fontsize=16, fontweight ='bold', loc='top')
    
    plt.ylim(0.05, 6.0)

    plt.legend(loc='upper right', prop={'size': 12})
    
    outputname = [f'{plotfoldername}/probability.png', f'{plotfoldername}/probability.pdf']
    plt.savefig(outputname[0])
    plt.savefig(outputname[1])
    print(f"Plotting the probability plot: {outputname[0]}")
    plt.close()

def roc_curve(y_value:dict, plotfoldername:str)->None:
    
    ax, fig = Setup_Plot_Style()
    
    # Train ROC
    fpr_train, tpr_train, _ = metrics.roc_curve(y_value['Train'][0].values, y_value['Train'][1].values[:, 1], pos_label=1)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    # Valid ROC
    fpr_valid, tpr_valid, _ = metrics.roc_curve(y_value['Valid'][0].values, y_value['Valid'][1].values[:, 1], pos_label=1)
    roc_auc_valid = metrics.auc(fpr_valid, tpr_valid)
    # Test ROC
    fpr_test, tpr_test, _ = metrics.roc_curve(y_value['Test'][0].values, y_value['Test'][1].values[:, 1], pos_label=1)
    roc_auc_test = metrics.auc(fpr_test, tpr_test)
    
    plt.plot(fpr_train, tpr_train, '#ff433d', label=f"AUC(Train) = {roc_auc_train:.3f}")
    plt.plot(fpr_valid, tpr_valid, '#4af6c3', label=f"AUC(Valid) = {roc_auc_valid:.3f}")
    plt.plot(fpr_test, tpr_test, '#0068ff', label=f"AUC(Test) = {roc_auc_test:.3f}")
    plt.legend(loc='lower right', fontsize=14)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel("Background Efficiency", fontsize=16, fontweight ='bold', loc='right')
    plt.xticks(fontsize=14)
    plt.ylabel("Signal Efficiency", fontsize=16, fontweight ='bold', loc='top')
    plt.yticks(fontsize=14)
    
    outputname = [f'{plotfoldername}/roc_curve.png', f'{plotfoldername}/roc_curve.pdf']
    plt.savefig(outputname[0])
    plt.savefig(outputname[1])
    print(f"Plotting the roc_curve plot: {outputname[0]}")
    plt.close()
    
def Fitting_Plots(loss_train:list, loss_test:list, plotfolder:str)->None:
    Setup_Plot_Style()
    
    folds = range(1, len(loss_train)+1)
    plt.plot(folds, loss_train, 'o-', color='green', label='train')
    plt.plot(folds, loss_test, 'o-', color='red', label='test')
    plt.xticks(folds) # Make the x-axis integer
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Number of fold', fontsize=16, loc='right')
    plt.ylabel('LogLoss', fontsize=16, loc='top')
    
    ymin = min([item for sublist in [loss_train, loss_test] for item in sublist]) * 0.95
    ymax = max([item for sublist in [loss_train, loss_test] for item in sublist]) * 1.05
    plt.ylim(ymin, ymax)
    
    plt.legend(loc='upper right', fontsize=14)
    plt.grid()
    outputname = [f'{plotfolder}/loss_train_test.png', f'{plotfolder}/loss_train_test.pdf']
    plt.savefig(outputname[0])
    plt.savefig(outputname[1])
    print(f"Plotting the loss_train_test plot: {outputname[0]}")
    plt.close()