import pandas as pd
import numpy as np

from scipy.stats import norm
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv('Datasets/train.csv')


# # ----------------------------------------#
# # First we test for teh Attribute 'Fare'
# # We will plot the values of 'Fare' to observe histogram and Normal Probability distribution
#
# fare = df_train.loc[df_train['Fare'] > 0, 'Fare']
# print(fare.describe())
# # Outliers can be observed
#
# sns.distplot(fare, fit=norm)
# fig = plt.figure()
# res = stats.probplot(fare, plot=plt)
# plt.show()
# # Observation: We observe a large deviation from the Normal curve
# # Solution: We will use Log Transformation to conform normality
#
# fare = np.log(df_train.loc[df_train['Fare'] > 0, 'Fare'])
#
# # Now we plot again to see the rectification
# sns.distplot(fare, fit=norm)
# fig = plt.figure()
# res = stats.probplot(fare, plot=plt)
# plt.show()
# print("Number of Fare values set to 0:", (len(df_train['Fare']) - len(fare)))
#
#
# # Now we perform Logistic regression with 'Survived', to determine the dependancy
# ss = StandardScaler()
# iv_s = ss.fit_transform(fare[:, np.newaxis])
# survived = np.array(df_train.loc[df_train['Fare'] > 0, 'Survived'].copy().apply(str))
# lr = LogisticRegression()
# lr.fit(iv_s.reshape(-1, 1), survived)
# print("Logistic Regression Coeff for Age:", lr.coef_[0])


# ---------------------------------------- #
# Now we analyse the variable Age
age = df_train['Age']
sur = df_train['Survived']
age_sur = pd.concat([age, sur], axis=1).dropna()
print(age)
age_sur.columns = ['Age', 'Survived']
print("Description for Age\n", age_sur['Age'].describe())

age_sur.loc[age_sur['Age'] < 1, 'Age':] *= 100
print("Description for Age\n", age_sur['Age'].describe())

sns.distplot(age_sur['Age'], fit=norm)
fig = plt.figure()
res = stats.probplot(age_sur['Age'], plot=plt)
plt.show()
# Observation: We do not observe a large deviation from the Normal curve

# Now we perform Logistic regression with 'Survived', to determine the dependancy
ss = StandardScaler()
iv_s = ss.fit_transform(age_sur['Age'][:, np.newaxis])
survived = np.array(age_sur['Survived'].copy().apply(str))
lr = LogisticRegression()
lr.fit(iv_s.reshape(-1, 1), survived)
print("Logistic Regression Coeff for Age:", lr.coef_[0])
print("Number of missing values:", (len(age) - age.count()))
print("-----\n")

#
# # ---------------------------------------- #
# # SibSp and Parch indicates number of family members on board.
# # We will sum them up and analyse its relation with Survived
# sibsp = df_train['SibSp']
# parch = df_train['Parch']
# sur = df_train['Survived']
#
# family_members = pd.DataFrame(sibsp + parch)
# fm_sur = pd.concat([family_members, sur], axis=1).dropna()
# fm_sur.columns = ['FamMem', 'Survived']
#
# fm_s = []
# fm_t = []
# fm = []
# for i in range(0, 11):
#     fm_s.append(len(fm_sur.loc[(fm_sur['FamMem'] == i) & (fm_sur['Survived'] == 1), 'Survived']))
#     v = len(fm_sur.loc[fm_sur['FamMem'] == i, 'Survived'])
#     if v == 0:
#         fm_t.append(99)
#     else:
#         fm_t.append(v)
#     fm.append(i)
# fm_s_np = np.array(fm_s)
# fm_t_np = np.array(fm_t)
# fm_s_r = (fm_s_np/fm_t_np) * 100
#
# print("Total Number of Family Members:", fm_t_np)
# print("In the above array 99 means 0. and was added for convenience for dividing")
# print("------\n")
#
# plt.bar(fm, fm_s_r)
# plt.xlabel('Total Family Members \n(Total Family Members 8 and 9 do not exist in the dataset)')
# plt.ylabel('Percentage Survivors')
# plt.show()
#
#
# # Observation: There is observed difference in survival with number of family members on board.
# # - No particular correlation can be observed.
#
#
# # ---------------------------------------- #
# # Now evaluating Role of Gender in Surviving
# sex = df_train['Sex']
# sur = df_train['Survived']
#
#
# male = pd.DataFrame(df_train.loc[df_train['Sex'] == 'male', 'Survived'])
# female = pd.DataFrame(df_train.loc[df_train['Sex'] == 'female', 'Survived'])
#
# d = {'Sex': ['male', 'female'], 'No. of Survivors': [len(male), len(female)]}
# sex_count_vs_sur = pd.DataFrame(data=d)
# plt.bar(sex_count_vs_sur['Sex'], sex_count_vs_sur['No. of Survivors'])
# plt.xlabel('Sex')
# plt.ylabel('No. Of Survivors')
# plt.show()
# # Observation: Among the survivors, there appears to be a higher number of men.
#
# female_s = df_train.loc[(df_train['Sex'] == 'female') & (df_train['Survived'] == 1), 'Survived']
# male_s = df_train.loc[(df_train['Sex'] == 'male') & (df_train['Survived'] == 1), 'Survived']
#
# female_sur_r = (len(female_s)/len(female))
# male_sur_r = (len(male_s)/len(male))
#
# plt.bar(['females', 'males'], [female_sur_r*100, male_sur_r*100])
# plt.ylabel('Percentage Survivors')
# plt.xlabel('Sex')
# plt.show()
# # Observation: Clearly, most of the females survived and very few men survived. There
# # - are only a few outliers
#
#
# # ---------------------------------------- #
# # Now evaluating Role of Gender in Surviving
#
# pclass_3 = pd.DataFrame(df_train.loc[df_train['Pclass'] == 3, 'Survived'])
# pclass_2 = pd.DataFrame(df_train.loc[df_train['Pclass'] == 2, 'Survived'])
# pclass_1 = pd.DataFrame(df_train.loc[df_train['Pclass'] == 1, 'Survived'])
#
# d = {'Pclass': ['3', '2', '1'], 'No. of Survivors': [len(pclass_3), len(pclass_2), len(pclass_1)]}
# pclass_count_vs_sur = pd.DataFrame(data=d)
# plt.bar(pclass_count_vs_sur['Pclass'], pclass_count_vs_sur['No. of Survivors'])
# plt.xlabel('Pclass')
# plt.ylabel('No. Of Survivors')
# plt.show()
# # Observation: Most survivors are from third class, around 500
#
# pclass3_s = df_train.loc[(df_train['Pclass'] == 3) & (df_train['Survived'] == 1), 'Survived']
# pclass2_s = df_train.loc[(df_train['Pclass'] == 2) & (df_train['Survived'] == 1), 'Survived']
# pclass1_s = df_train.loc[(df_train['Pclass'] == 1) & (df_train['Survived'] == 1), 'Survived']
#
# pclass3_sur_r = (len(pclass3_s)/len(pclass_3))
# pclass2_sur_r = (len(pclass2_s)/len(pclass_2))
# pclass1_sur_r = (len(pclass1_s)/len(pclass_1))
#
# plt.bar(['1', '2', '3'], [pclass1_sur_r*100, pclass2_sur_r*100, pclass3_sur_r*100])
# plt.ylabel('Percentage Survivors')
# plt.xlabel('Pclass')
# plt.show()
# # Observation: Many of the 1st class passengers survived
#
