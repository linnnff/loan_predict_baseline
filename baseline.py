import pandas as pd
import lightgbm as lgb
import numpy as np

train_data = pd.read_csv('daik/input/train_public.csv')
train_internet = pd.read_csv('daik/input/train_internet.csv')
submit_example = pd.read_csv('daik/input/submit_example.csv')
test_public = pd.read_csv('daik/input/test_public.csv')

work_year_dict = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10,
}

train_data['work_year'] = train_data['work_year'].map(work_year_dict)
test_public['work_year'] = test_public['work_year'].map(work_year_dict)
train_data['work_year'] = train_data['work_year'].fillna(-1)
test_public['work_year'] = test_public['work_year'].fillna(-1)


class_dict = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

train_data['class'] = train_data['class'].map(class_dict)
test_public['class'] = test_public['class'].map(class_dict)

train_data['issue_date'] = pd.to_datetime(train_data['issue_date'])
test_public['issue_date'] = pd.to_datetime(test_public['issue_date'])


train_data['issue_date_month'] = train_data['issue_date'].dt.month
test_public['issue_date_month'] = train_data['issue_date'].dt.month

train_data['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek
test_public['issue_date_dayofweek'] = train_data['issue_date'].dt.dayofweek


col_to_drop = ['issue_date', 'earlies_credit_mon']
train_data = train_data.drop(col_to_drop, axis=1)
test_public = test_public.drop(col_to_drop, axis=1)

cat_cols = ['employer_type', 'industry']

from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    lbl = LabelEncoder().fit(train_data[col])
    train_data[col] = lbl.transform(train_data[col])
    test_public[col] = lbl.transform(test_public[col])
# print(train_data[cat_cols].head())
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# print(train_data.head())
import matplotlib.pyplot as plt
import seaborn as sns


def k_fold_serachParmaters(model,train_data, train_label, test_data):
    n_splits=5
    
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2020)
    pred_Test = np.zeros(len(test_data))
    
    auc_train, auc_val = 0, 0
    for tr_idx, val_idx in sk.split(train_data, train_label):
        x_train = train_data.iloc[tr_idx]
        y_train = train_label.iloc[tr_idx]
        x_val = train_data.iloc[val_idx]
        y_val = train_label.iloc[val_idx]

        model.fit(x_train, y_train, 
                  eval_set=[(x_val, y_val)], 
                  categorical_feature = cat_cols,
                 early_stopping_rounds=100,
                 verbose=False)

        pred_Test += model.predict_proba(test_data)[:, 1]/n_splits

        pred = model.predict(x_val)
        auc_val += roc_auc_score(y_val,pred)/n_splits
        
        pred = model.predict(x_train)
        auc_train += roc_auc_score(y_train, pred)/n_splits
        
        
    return auc_val, pred_Test

import warnings
warnings.filterwarnings("ignore")

score_tta = None
score_list = []

tta_fold = 20
for _ in range(tta_fold):
    clf = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=32,
                           max_depth=6,
                           learning_rate=0.1,
                           n_estimators=10000,
                           subsample=0.8,
                           feature_fraction=0.6,
                           reg_alpha=0.5,
                           reg_lambda=0.5,
                           random_state=2021,
                           is_unbalance=True,
                           metric='auc')

    score, test_pred = k_fold_serachParmaters(clf,
                           train_data.drop(['loan_id', 'user_id', 'isDefault'], axis=1),
                           train_data['isDefault'],
                           test_public.drop(['loan_id', 'user_id',], axis=1),
                          )

    print(score)
    if score_tta is None:
        score_tta = test_pred/tta_fold
    else:
        score_tta += test_pred/tta_fold
    score_list.append(score)

test_public['isDefault'] = score_tta
test_public.rename({'loan_id': 'id'}, axis=1)[['id', 'isDefault']].to_csv('aaa.csv', index=None)