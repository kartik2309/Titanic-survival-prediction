import pandas as pd
from numpy import log
from numpy import newaxis
from sklearn.preprocessing import StandardScaler


def get_clean_data():
    path = 'Datasets/train.csv'
    df_train = pd.read_csv(path)

    # -----------------------------------#
    # Clean the variable Fare.
    # Setting it 0's equal to median of non-zeros
    fare = df_train['Fare']
    fare.replace(0, fare.median(), inplace=True)
    fare_log = log(fare)
    ss = StandardScaler()
    fare_log_ss = pd.DataFrame(ss.fit_transform(fare_log[:, newaxis]))
    fare_log_ss.columns = ['Fare']

    # -----------------------------------#
    # Clean the variable SibSp, Parch.
    sibsp = df_train['SibSp']
    parch = df_train['Parch']
    family_members = pd.DataFrame(sibsp + parch)
    family_members.columns = ['FamMem']
    family_members_d = pd.get_dummies(family_members['FamMem']).reindex(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fill_value=0)
    family_members_d.drop(family_members_d.columns[0])

    # -----------------------------------#
    # Clean the variable Sex.
    sex = df_train['Sex']
    sex_d = pd.get_dummies(sex).reindex(columns=['male', 'female'])
    sex_d.drop(sex_d.columns[0], inplace=True, axis=1)

    # -----------------------------------#
    # Clean the variable Pclass.
    pclass = df_train['Pclass']
    pclass_d = pd.get_dummies(pclass).reindex(columns=[1, 2, 3])
    pclass_d.drop(pclass_d.columns[0], inplace=True, axis=1)

    df_clean = pd.concat([fare_log_ss, sex_d, pclass_d,
                          df_train['Survived'].apply(str)], axis=1)
    return df_clean


def get_clean_test_data():
    path = 'Datasets/test.csv'
    df_test = pd.read_csv(path)
    # -----------------------------------#
    # Clean the variable Fare.
    # Setting it 0's equal to median of non-zeros
    fare = df_test['Fare']
    fare.replace(0, fare.median(), inplace=True)
    fare.fillna(fare.median(), inplace=True)
    fare_log = log(fare)
    ss = StandardScaler()
    fare_log_ss = pd.DataFrame(ss.fit_transform(fare_log[:, newaxis]))
    fare_log_ss.columns = ['Fare']

    # -----------------------------------#
    # Clean the variable SibSp, Parch.
    sibsp = df_test['SibSp']
    parch = df_test['Parch']
    family_members = pd.DataFrame(sibsp + parch)
    family_members.columns = ['FamMem']
    family_members_d = pd.get_dummies(family_members['FamMem']).reindex(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                                        fill_value=0)
    family_members_d.drop(family_members_d.columns[0])

    # -----------------------------------#
    # Clean the variable Sex.
    sex = df_test['Sex']
    sex_d = pd.get_dummies(sex).reindex(columns=['male', 'female'])
    sex_d.drop(sex_d.columns[0], inplace=True, axis=1)

    # -----------------------------------#
    # Clean the df_test Pclass.
    pclass = df_test['Pclass']
    pclass_d = pd.get_dummies(pclass).reindex(columns=[1, 2, 3])
    pclass_d.drop(pclass_d.columns[0], inplace=True, axis=1)

    df_clean = pd.concat([fare_log_ss, sex_d, pclass_d], axis=1)
    return df_clean


def get_id_from_test():
    path = 'Datasets/test.csv'
    df_test = pd.read_csv(path)

    ids = pd.DataFrame(df_test['PassengerId'])
    ids.columns = ['PassengerId']

    return ids
