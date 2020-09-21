import pandas as pd
import matplotlib.pyplot as plt

df_public = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent',
                                 'Age1stCode',
                                 'YearsCode',
                                 'YearsCodePro',
                                 'ConvertedComp',
                                 'Age'
                                 # 'BetterLife',
                                 # 'JobSat'
                                 ],
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

# change all to float
df_public = df_public.astype(float)

print(df_public.corr())

plt.plot(df_public['YearsCodePro'],
         df_public['ConvertedComp'], 'ro', markersize=0.3)
plt.xlabel('YearsCodePro')
plt.ylabel('ConvertedComp')
plt.title('All')
plt.show()
