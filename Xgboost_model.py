import xgboost as xgb
from sklearn.model_selection import train_test_split


class Xgboost_model:

    def model(self, data, label, feature_name):

        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

        dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_name)
        dtest = xgb.DMatrix(X_test, y_test, feature_names=feature_name)

        print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
        print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))

        params = {
            'objective': 'binary:logistic',
            'max_depth': 7,
            'max_delta_step': 1,
            'scale_pos_weight': 3

        }

        num_rounds = 70
        evallist = [(dtest, 'eval'), (dtrain, 'train')]

        model = xgb.train(params, dtrain, num_rounds, evals=evallist)

        return model

