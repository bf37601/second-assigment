import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error


df_public = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent',
                                 'Age1stCode',
                                 'YearsCode',
                                 'YearsCodePro',
                                 'ConvertedComp',
                                 'Age',
                                 'BetterLife',
                                 'JobSat'],
                        index_col='Respondent')
df_public.dropna(inplace=True)

# replace string values
df_public.replace(to_replace='Less than 1 year', value='0', inplace=True)
df_public.replace(to_replace='More than 50 years', value='51', inplace=True)
df_public.replace(to_replace='Younger than 5 years', value='4', inplace=True)
df_public.replace(to_replace='1e+05', value='100000', inplace=True)
df_public.replace(to_replace='1e+06', value='1000000', inplace=True)
df_public.replace(to_replace='2e+05', value='200000', inplace=True)
df_public.replace(to_replace='2e+06', value='2000000', inplace=True)
df_public.replace(to_replace='3e+05', value='300000', inplace=True)
df_public.replace(to_replace='4e+05', value='400000', inplace=True)
df_public.replace(to_replace='6e+05', value='600000', inplace=True)
df_public.replace(to_replace='9e+05', value='900000', inplace=True)
df_public.replace(to_replace="Yes", value='1', inplace=True)
df_public.replace(to_replace="No", value='0', inplace=True)

onehotencoder = LabelEncoder()
df_public['JobSat'] = onehotencoder.fit_transform(df_public['JobSat'])

# change all to float
df_public = df_public.astype(float)


# y = Age
# x1 = YearsCodePro
# x2 = ConvertedComp

# +-3 standard deviation elimination
df_public_sd = df_public[np.abs(df_public - df_public.mean()) <= 3*df_public.std()]
df_public_sd.isna().sum()
df_public_sd = df_public_sd.dropna()

# quantile elimination
Q1 = df_public.quantile(0.25)
Q3 = df_public.quantile(0.75)
IQR = Q3 - Q1
df_public_q = df_public[~((df_public < (Q1 - 1.5 * IQR)) | (df_public > (Q3 + 1.5 * IQR))).any(axis=1)]

pd.set_option('display.max_columns', None)

print(df_public.corr())
print(df_public_sd.corr())
print(df_public_q.corr())

df_public.plot()
plt.show()
df_public_sd.plot()
plt.show()
df_public_q.plot()
plt.show()

sns.boxplot(y='ConvertedComp', data=df_public_sd)
plt.show()
sns.boxplot(y='ConvertedComp', data=df_public_q)
plt.show()
sns.boxplot(y='YearsCodePro', data=df_public_sd)
plt.show()
sns.boxplot(y='YearsCodePro', data=df_public_q)
plt.show()
sns.boxplot(y='Age', data=df_public_sd)
plt.show()
sns.boxplot(y='Age', data=df_public_q)
plt.show()


reg = linear_model.LinearRegression()

df_len = len(df_public_q)
y_true = df_public_q['Age']


# x1
reg.fit(df_public_q[['YearsCodePro']], df_public_q['Age'])

# x2
# reg.fit(df_public_q[['ConvertedComp']], df_public_q['Age'])

y_test = pd.DataFrame(np.random.rand(df_len)*100)  # for x1 or x2

# x1 & x2
# reg.fit(df_public_q[['YearsCodePro', 'ConvertedComp']], df_public_q['Age'])
# y_test = pd.DataFrame(np.random.rand(df_len,2)*100)

# x1 & x2 & 'BetterLife' & 'JobSat'
# reg.fit(df_public_q[['YearsCodePro', 'ConvertedComp', 'BetterLife', 'JobSat']], df_public_q['Age'])
# y_test = pd.DataFrame(np.random.rand(df_len,4)*100)

y_pred = reg.predict(y_test)
print(mean_squared_error(y_true, y_pred))


