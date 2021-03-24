
import xgboost as xgb
from project.utils import load_data
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score,confusion_matrix,classification_report, roc_curve, auc, plot_roc_curve,brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import scikitplot as skplt
import pickle as pkl
data_dir = r'data/data_new_models.xlsx'

datasets = [
    'blanco_cac',
    'blanco',
    'non_blanco'
]

pbounds = {
    'learning_rate': (0.01, 0.1),
    'n_estimators': (100, 400),
    'max_depth': (3, 5),
    'subsample': (0.6, 0.9),  # Change for big datasets
    'colsample_bytree': (0.6, 0.9),  # Change for datasets with lots of features
    'gamma': (0, 5),
}


def print_model_results(model,x,y):
    # calibrator = CalibratedClassifierCV(model,cv='prefit')
    # calibrator.fit(x, y)

    y_pred = np.round(model.predict(x))
    y_scores = model.predict_proba(x)[:, 1]
    print('regular_results______:')
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))
    fpr, tpr, auc_thresholds = roc_curve(y, y_scores)
    print('AUC: {}'.format(auc(fpr, tpr))) # AUC of ROC
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.plot(fpr, tpr, linewidth=1.5, label=item)


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1 - f1_score(y_true, np.round(y_pred))
    return 'f1_err', err

def xgboost_hyper_param(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        colsample_bytree,
                        gamma,
                        ):
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
        eta=0.6
        )
    return np.mean(cross_val_score(clf, x_train, y_train, cv=5, scoring='f1'))



optimized_params={}
for item in datasets:
    x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, item)
    print('Optimization starting for current dataset:{}'.format(item))

    optimizer = BayesianOptimization(
        f=xgboost_hyper_param,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    optimizer.maximize(n_iter=30)

    print(optimizer.max['target'])
    params = optimizer.max['params']
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])

    optimized_params[item]=params

    cardiacXGB = xgb.XGBClassifier(eval_metric='auc')
    cardiacXGB.set_params(**params)

    cardiacXGB.fit(x_train,y_train)

    print('Training set for {} results:'.format(item))
    print_model_results(cardiacXGB, x_train, y_train)
    print('Test set  for {} results:'.format(item))
    print_model_results(cardiacXGB, x_test, y_test)

optimized_params

# Model Training performance
optimized_params =  {'blanco_cac': {'colsample_bytree': 0.8802557538761343, 'gamma': 4.238243901343139, 'learning_rate': 0.05883147793278661, 'max_depth': 3, 'n_estimators': 231, 'subsample': 0.8182246448121042}, 'blanco': {'colsample_bytree': 0.9989855145520636, 'gamma': 3.422172846756493, 'learning_rate': 0.08868722561458804, 'max_depth': 6, 'n_estimators': 384, 'subsample': 0.9195140809945592}, 'non_blanco': {'colsample_bytree': 0.8528719724947116, 'gamma': 4.702433457052847, 'learning_rate': 0.09021691676548928, 'max_depth': 3, 'n_estimators': 218, 'subsample': 0.8696808199613849}}
optimized_params= {'blanco_cac': {'colsample_bytree': 0.6421160815785701, 'gamma': 0.990507445424394, 'learning_rate': 0.0820670111807983, 'max_depth': 4, 'n_estimators': 194, 'subsample': 0.8076967847007942}, 'blanco': {'colsample_bytree': 0.6277599256724903, 'gamma': 4.968398554039898, 'learning_rate': 0.051645492829398, 'max_depth': 3, 'n_estimators': 188, 'subsample': 0.6349531465914797}, 'non_blanco': {'colsample_bytree': 0.7450212341182476, 'gamma': 3.8019087108564893, 'learning_rate': 0.08299357009729698, 'max_depth': 4, 'n_estimators': 362, 'subsample': 0.8136203789251157}}


outputfolder = r'C:\Users\rut-g\PycharmProjects\cardiowebML\models'
import os

for dataset in datasets:
    x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, dataset)


    params = optimized_params[dataset]
    print(params)
    cardiacXGB = xgb.XGBClassifier()
    cardiacXGB.set_params(**params)
    eval_set = [(x_test, y_test)]

    cardiacXGB.fit(x_train,y_train,early_stopping_rounds=10, eval_set=eval_set, eval_metric='auc')
    pkl.dump(cardiacXGB,open(os.path.join(outputfolder,dataset), 'wb'))

plt.close('all')

for model in os.listdir(outputfolder):
    if 'blanco_cac' in model:
        x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[0])
    elif 'non_blanco' in model:
        x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[2])
    else:
        x_train, y_train, x_test, y_test, key_train, key_test = load_data(data_dir, datasets[1])


    print(model)
    xgb_model = pkl.load(open(os.path.join(outputfolder,model),'rb'))

    print('-----  train results  -----')
    print_model_results(xgb_model, x_train,y_train)

    print('-----  test results  -----')
    print_model_results(xgb_model, x_test, y_test)



