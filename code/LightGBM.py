from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from dealdatafile import feature
import numpy as np
import gc

num_fold = 5

if __name__ == '__main__':
    kf = KFold(n_splits=num_fold, shuffle=False, random_state=1)
    error = 0
    models = []
    train = pd.read_csv('ashrae-energy-prediction/train_cleaned.csv')
    target = np.log1p(train['meter_reading'])
    train = train.drop('meter_reading', axis=1)
    train['square_feet'] = np.log1p(train['square_feet'])
    train['year_built'] = train['year_built'] - 1900
    for i, (train_index, val_index) in enumerate(kf.split(train)):
        if i + 1 < num_fold:
            continue
        print(train_index.max(), train_index.min())
        print(val_index.max(), val_index.min())
        train_X = train[feature].iloc[train_index]
        val_X = train[feature].iloc[val_index]
        train_Y = target.iloc[train_index]
        val_Y = target.iloc[val_index]
        lgb_train = lgb.Dataset(train_X, train_Y > 0)
        lgb_eval = lgb.Dataset(val_X, val_Y > 0)

        param = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss'},
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }

        gbm_class = lgb.train(param, lgb_train, num_boost_round=2000, valid_sets=(lgb_train, lgb_eval),
                              early_stopping_rounds=20, verbose_eval=20)

        lgb_train = lgb.Dataset(train_X[train_Y > 0], train_Y[train_Y > 0])
        lgb_eval = lgb.Dataset(val_X[val_Y > 0], val_Y[val_Y > 0])

        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5
        }

        gbm_regress = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=(lgb_train, lgb_eval),
                              early_stopping_rounds=20, verbose_eval=20)

        y_pred = (gbm_class.predict(val_X, num_iteration=gbm_class.best_iteration) > .5) * \
                 (gbm_regress.predict(val_X, num_iteration=gbm_regress.best_iteration))

        error += np.sqrt(mean_squared_error(y_pred, (val_Y))) / num_fold
        print(np.sqrt((mean_squared_error(y_pred, (val_Y)))))
        break
    print(error)

    gbm_class.save_model('LightGBM/gbm_class_log_square_feet.model')
    gbm_regress.save_model('LightGBM/gbm_regress_log_square_feet.model')

    # sorted(zip(gbm_regress.feature_importance(), gbm_regress.feature_name()), reverse=True)
    # del train
    # del train_X, train_Y, lgb_train, lgb_eval, val_Y, val_X, y_pred, target
    # gc.collect()
    #
    # test = pd.read_csv("ashrae-energy-prediction/test.csv")
    # building_info = pd.read_csv("ashrae-energy-prediction/building_metadata.csv")
    # test = test.merge(building_info, left_on="building_id", right_on="building_id", how="left")
    # del building_info
    # gc.collect()
