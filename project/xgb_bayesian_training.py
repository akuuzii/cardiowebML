import os
import xgboost as xgb
from project.utils import load_data
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,confusion_matrix,classification_report, roc_curve, auc, plot_roc_curve,brier_score_loss
import matplotlib.pyplot as plt
import scikitplot as skplt
import pickle as pkl
import pandas as pd
from imblearn.over_sampling import ADASYN


data_dir = r'data/data_new_models.xlsx'

datasets = [
    'blanco_cac',
    'blanco',
    'non_blanco'
]

pbounds = {
    'learning_rate': (0.01, 0.1),
    'n_estimators': (100, 150),
    'max_depth': (3, 4),
    'subsample': (0.5, 0.6),  # Change for big datasets
    'colsample_bytree': (0.5, 0.6),  # Change for datasets with lots of features
    'gamma': (0, 5),
    'scale_pos_weight': (1,9),
    'eta':(0.5,0.7)
}

def _F1_eval(preds, labels):
    t = np.arange(0, 1, 0.005)
    f = np.repeat(0, 200)
    results = np.vstack([t, f]).T
    # assuming labels only containing 0's and 1's
    n_pos_examples = sum(labels)
    if n_pos_examples == 0:
        raise ValueError("labels not containing positive examples")

    for i in range(200):
        pred_indexes = (preds >= results[i, 0])
        TP = sum(labels[pred_indexes])
        FP = len(labels[pred_indexes]) - TP
        precision = 0
        recall = TP / n_pos_examples

        if (FP + TP) > 0:
            precision = TP / (FP + TP)

        if (precision + recall > 0):
            F1 = 2 * precision * recall / (precision + recall)
        else:
            F1 = 0
        results[i, 1] = F1
    return (max(results[:, 1]))

def F1_eval(preds, dtrain):
    res = _F1_eval(preds, dtrain.get_label())
    return 'f1_err', 1-res





def xgboost_hyper_param(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        colsample_bytree,
                        gamma,
                        scale_pos_weight,
                        eta):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    clf = xgb.XGBClassifier(
        verbosity=0,
        objective='binary:logistic',
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        scale_pos_weight=scale_pos_weight,
        eta=eta

        )
    return np.mean(cross_val_score(clf, x_train, y_train, cv=10, scoring='f1'))



optimized_params={}
for item in datasets:
    x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, item)
    # ada = ADASYN(random_state=42)
    # x_train, y_train =  ada.fit_resample(x_train, y_train)



    print('Optimization starting for current dataset:{}'.format(item))

    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
        verbose=2,
    )

    optimizer.maximize(n_iter=30)

    print(optimizer.max['target'])
    params = optimizer.max['params']
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    optimized_params[item]=params



# optimized_params

# Model Training performance
# optimized_params =  {'blanco_cac': {'colsample_bytree': 0.8802557538761343, 'gamma': 4.238243901343139, 'learning_rate': 0.05883147793278661, 'max_depth': 3, 'n_estimators': 231, 'subsample': 0.8182246448121042}, 'blanco': {'colsample_bytree': 0.9989855145520636, 'gamma': 3.422172846756493, 'learning_rate': 0.08868722561458804, 'max_depth': 6, 'n_estimators': 384, 'subsample': 0.9195140809945592}, 'non_blanco': {'colsample_bytree': 0.8528719724947116, 'gamma': 4.702433457052847, 'learning_rate': 0.09021691676548928, 'max_depth': 3, 'n_estimators': 218, 'subsample': 0.8696808199613849}}
# optimized_params= {'blanco_cac': {'colsample_bytree': 0.6421160815785701, 'gamma': 0.990507445424394, 'learning_rate': 0.0820670111807983, 'max_depth': 4, 'n_estimators': 194, 'subsample': 0.8076967847007942}, 'blanco': {'colsample_bytree': 0.6277599256724903, 'gamma': 4.968398554039898, 'learning_rate': 0.051645492829398, 'max_depth': 3, 'n_estimators': 188, 'subsample': 0.6349531465914797}, 'non_blanco': {'colsample_bytree': 0.7450212341182476, 'gamma': 3.8019087108564893, 'learning_rate': 0.08299357009729698, 'max_depth': 4, 'n_estimators': 362, 'subsample': 0.8136203789251157}}
# # new params
# optimized_params = {'blanco_cac': {'colsample_bytree': 0.7253268339364138, 'gamma': 4.327874980017729, 'learning_rate': 0.02458136475045658, 'max_depth': 3, 'n_estimators': 290, 'scale_pos_weight': 5.500841005283995, 'subsample': 0.6636534163868724}, 'blanco': {'colsample_bytree': 0.7132603485653572, 'gamma': 4.017795496567467, 'learning_rate': 0.012509547620182252, 'max_depth': 4, 'n_estimators': 292, 'scale_pos_weight': 5.2665377344707345, 'subsample': 0.605299420998397}, 'non_blanco': {'colsample_bytree': 0.6054775186395852, 'gamma': 3.352337550892011, 'learning_rate': 0.047557432213041435, 'max_depth': 4, 'n_estimators': 128, 'scale_pos_weight': 2.5848119126790303, 'subsample': 0.7601489137351074}}
# optimized_params={'blanco_cac': {'colsample_bytree': 0.6, 'eta': 0.5, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 194, 'scale_pos_weight': 2.0, 'subsample': 0.8}, 'blanco': {'colsample_bytree': 0.8, 'eta': 0.6, 'gamma': 0.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 197, 'scale_pos_weight': 2.0, 'subsample': 0.8}, 'non_blanco': {'colsample_bytree': 0.7537656474033627, 'eta': 0.5385680988350954, 'gamma': 1.4628562947153072, 'learning_rate': 0.07981828148257573, 'max_depth': 4, 'n_estimators': 165, 'scale_pos_weight': 1.8567782861865445, 'subsample': 0.7383410939441257}}

outputfolder = r'C:\Users\rut-g\PycharmProjects\cardiowebML\models'



for dataset in datasets:
    x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, dataset)


    params = optimized_params[dataset]
    cardiacXGB = xgb.XGBClassifier()
    cardiacXGB.set_params(**params)
    eval_set = [(x_test, y_test)]

    cardiacXGB.fit(x_train,y_train,early_stopping_rounds=10, eval_set=eval_set,eval_metric=F1_eval) #,
    pkl.dump(cardiacXGB,open(os.path.join(outputfolder,dataset), 'wb'))

plt.close('all')



models = r'C:\Users\rut-g\PycharmProjects\cardiowebML\models_final'
with pd.ExcelWriter('output.xlsx') as writer:

    for index, model in enumerate(os.listdir(models)):
        if 'blanco_cac' in model:
            x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[0])
        elif 'non_blanco' in model:
            x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[2])
        else:
            x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[1])




        xgb_model_1 = pkl.load(open(os.path.join(models, model), 'rb'))

        xgb.plot_importance(xgb_model_1,title=model)



        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].set_title('{}'.format(model))
        axs[0].set(xlabel=r'$\bf{1}$ $\bf{-}$ $\bf{Specificity}$', ylabel=r'$\bf{Sensitivity}$')
        axs[0].plot([0, 1], [0, 1], 'k--')
        axs[0].axis([-0.005, 1, 0, 1.005])

        for set in [1, 2]:
            if set == 1:
                x = x_train
                y = y_train.array
                label = 'Training set'
                y_scores = xgb_model_1.predict_proba(x)[:, 1]
                y_pred = np.round(y_scores)
                skplt.metrics.plot_confusion_matrix(y, y_pred, ax=axs[1],normalize=False,title='Training dataset results')



            else:
                x = x_test
                y = y_test.array
                label = 'Testing set'
                y_scores = xgb_model_1.predict_proba(x)[:, 1]
                y_pred = np.round(y_scores)
                skplt.metrics.plot_confusion_matrix(y, y_pred, ax=axs[2],normalize=False,title='Test dataset results')

                indices_FN = [i for i in range(len(y_test)) if y_test.iloc[i] == 1 and y_test.iloc[i] != y_pred[i]]

                indices_TP = [i for i in range(len(y_test)) if y_test.iloc[i] == 1 and y_test.iloc[i] == y_pred[i]]
                #
                indices_TN = [i for i in range(len(y_test)) if y_test.iloc[i] == 0 and y_test.iloc[i] == y_pred[i]]

                indices_FP = [i for i in range(len(y_test)) if y_test.iloc[i] == 0 and y_test.iloc[i] != y_pred[i]]

                false_negatives = x_test.iloc[indices_FN, :].mean()
                true_positives = x_test.iloc[indices_TP, :].mean()
                true_negatives = x_test.iloc[indices_TN, :].mean()
                false_positives = x_test.iloc[indices_FP, :].mean()

                df = pd.DataFrame(data=[false_negatives, true_positives, true_negatives, false_positives],
                                  index=['False Negatives', 'True Positives', 'True Negatives', 'False Positives'],
                                  columns=x_test.columns)

                df.T.to_excel(writer, sheet_name=model)
            # print(confusion_matrix(y, y_pred))
            # print(classification_report(y, y_pred))
            fpr, tpr, auc_thresholds = roc_curve(y, y_scores)
            axs[0].plot(fpr,tpr,label=label + ' (AUC: '+str(round(auc(fpr, tpr), 3)) +')')

        legend = axs[0].legend(frameon=False,title=r'                    $\bf{AUC}$')
        renderer = fig.canvas.get_renderer()
        shift = max([t.get_window_extent(renderer).width for t in legend.get_texts()])
        for t in legend.get_texts():
            t.set_ha('right') # ha is alias for horizontalalignment
            t.set_position((shift,0))





def print_model_results(model,x,y):
    y_pred = np.round(model.predict(x))
    y_scores = model.predict_proba(x)[:, 1]

    fpr, tpr, auc_thresholds = roc_curve(y, y_scores)

    print('regular_results______:')
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    print('AUC: {}'.format(auc(fpr, tpr))) # AUC of RO

    # skplt.metrics.plot_roc(y, y_scores, title='ROC Curves', plot_micro=True, plot_macro=True, classes_to_plot=None, ax=None, figsize=None, cmap='nipy_spectral', title_fontsize='large', text_fontsize='medium')
    # skplt.metrics.plot_calibration_curve(y, [y_scores], clf_names=['xgb'])
    brier_loss = brier_score_loss(y,y_scores)
    print('Brier loss: {}'.format(brier_loss))

    skplt.metrics.plot_confusion_matrix(y, y_pred, normalize=False)




with pd.ExcelWriter('output.xlsx') as writer:

    for model in os.listdir(outputfolder):
        if 'blanco_cac' in model:
            x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[0])
        elif 'non_blanco' in model:
            x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[2])
        else:
            x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[1])

        print(model)
        xgb_model = pkl.load(open(os.path.join(outputfolder,model),'rb'))
        plt.figure()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].title('{}'.format(model))
        axs[0].set(xlabel=r'$\bf{1}$ $\bf{-}$ $\bf{Specificity}$',ylabel=r'$\bf{Sensitivity}$')
        axs[0].plot([0, 1], [0, 1], 'k--')
        axs[0].axis([-0.005, 1, 0, 1.005])
        for set in [1,2]:
            if set == 1:
                x = x_train
                y = y_train.array
                label = 'Training set'
            else:
                x = x_test
                y = y_test.array
                label = 'Testing set'

            y_scores = xgb_model.predict_proba(x)[:,1]
            fpr, tpr, auc_thresholds = roc_curve(y, y_scores)
            print(len(fpr))
            axs[0].plot(fpr,tpr,label=label + ' (AUC: '+str(round(auc(fpr, tpr),3)) +')')
            axs[0].legend()







