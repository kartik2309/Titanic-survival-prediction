from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from data_cleaning import get_clean_data
from data_cleaning import get_clean_test_data
from data_cleaning import get_id_from_test

from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def ensemble_model(X_train, y_train, X_test):
    # ------------------------------------  #

    svc = SVC(kernel='rbf')
    svc.fit(X_train, y_train)
    s_pred_svc = pd.DataFrame(svc.predict(X_test))

    # ------------------------------------ #

    rfc = RandomForestClassifier(n_estimators=6)
    rfc.fit(X_train, y_train)
    s_pred_rfc = pd.DataFrame(rfc.predict(X_test))

    # ------------------------------------ #

    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    s_pred_gbc = pd.DataFrame(gbc.predict(X_test))

    # ------------------------------------ #

    mlpc = MLPClassifier()
    mlpc.fit(X_train, y_train)
    s_pred_mlpc = pd.DataFrame(mlpc.predict(X_test))

    # ------------------------------------ #

    bgc = BaggingClassifier()
    bgc.fit(X_train, y_train)
    s_pred_bgc = pd.DataFrame(bgc.predict(X_test))

    # ------------------------------------ #

    s_pred_df = pd.concat([s_pred_svc, s_pred_rfc, s_pred_gbc, s_pred_mlpc, s_pred_bgc], axis=1)
    s_pred = s_pred_df.mode(axis=1)
    return s_pred

# Test code for our training dataset
# df = get_clean_data()
# y = pd.DataFrame(df['Survived'])
# y.columns = ['Survived']
# X = df.loc[:, df.columns != 'Survived']
# X_train, X_test, y_train, y_test = np.array(train_test_split(X, y, test_size=0.2, random_state=42))
#
# y_pred = ensemble_model(X_train, y_train, X_test)
# accs = accuracy_score(y_test, y_pred)
# print(accs)


# Main code to predict Survived from test.csv
df_train = get_clean_data()
X_test = get_clean_test_data()

y_train = pd.DataFrame(df_train['Survived'])
y_train.columns = ['Survived']
X_train = df_train.loc[:, df_train.columns != 'Survived']

survived = ensemble_model(X_train, y_train, X_test)
ids = get_id_from_test()
predictions = pd.concat([ids, survived], axis=1)
predictions.to_csv('predictions.csv', header=['PassengerId', 'Survived'], index=False)
print(predictions)
