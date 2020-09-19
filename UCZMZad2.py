import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

df_public = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent',
                                 'YearsCodePro',
                                 'Age'],
                        index_col='Respondent')
df_public.dropna(inplace=True)

# replace string values
df_public.replace(to_replace='Less than 1 year', value='0', inplace=True)

# change all to float
df_public = df_public.astype(float)

# +-3 standard deviation elimination
df_public_sd = df_public[np.abs(df_public - df_public.mean()) <= 3*df_public.std()]
df_public_sd.isna().sum()
df_public_sd = df_public_sd.dropna()

pd.set_option('display.max_columns', None)
print(df_public.corr())
print(df_public_sd.corr())

plt.scatter(df_public['YearsCodePro'], df_public['Age'])
plt.show()
plt.scatter(df_public_sd['YearsCodePro'], df_public_sd['Age'])
plt.show()

df_public.plot()
plt.show()
df_public_sd.plot()
plt.show()

sns.boxplot(y='YearsCodePro', data=df_public)
plt.show()
sns.boxplot(y='YearsCodePro', data=df_public_sd)
plt.show()

# linear regression
reg = linear_model.LinearRegression()

df_len = len(df_public_sd)
y_true = df_public_sd['Age']

reg.fit(df_public_sd[['YearsCodePro']], df_public_sd['Age'])
y_test = pd.DataFrame(np.random.rand(df_len)*100)
y_pred = reg.predict(y_test)

# MSE
print(mean_squared_error(y_true, y_pred))
