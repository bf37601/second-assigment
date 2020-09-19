import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

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

pd.set_option('display.max_columns', None)
print(df_public.corr())
