import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class XGBoost_model:

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

     def predict(self, model, dtest):

        predict = model.predict(dtest, ntree_limit=model.best_ntree_limit)

        return predict

    def predict_info(self, model, predict, dtest):

        predicted_labels = predict > 0.5

        print('Accuracy: {0:.2f}'.format(accuracy_score(dtest.get_label(), predicted_labels)))
        print('Precision: {0:.2f}'.format(precision_score(dtest.get_label(), predicted_labels)))
        print('Recall: {0:.2f}'.format(recall_score(dtest.get_label(), predicted_labels)))
        print('F1: {0:.2f}'.format(f1_score(dtest.get_label(), predicted_labels)))

        importances = model.get_fscore()
        print(importances)


