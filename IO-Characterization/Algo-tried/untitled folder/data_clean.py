import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.float_format', '{:.10f}'.format)



df = pd.read_parquet('argonne-full.parquet', engine='pyarrow')

df = df.iloc[:, 3:]
df = df.drop('MACHINE_NAME', axis=1)

df.head().T.to_csv('OUT.csv')

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
# print(IQR)

cols = list(df.columns)

for i in range(len(cols)):

    print(cols[i], ': {:.3f}'.format(df[cols[i]].skew()), end='  >>>>  ')

    q1 = df[cols[i]].quantile(0.10)
    q2 = df[cols[i]].quantile(0.90)
    
    df[cols[i]] = np.where(df[cols[i]] < q1, q1, df[cols[i]])
    df[cols[i]] = np.where(df[cols[i]] > q2, q2, df[cols[i]])

    print('{:.3f}'.format(df[cols[i]].skew()))


df.head().T.to_csv('OUT1.csv')


df = pd.read_parquet('argonne-full.parquet', engine='pyarrow')
df = df.iloc[:, 3:]
df = df.drop('MACHINE_NAME', axis=1)
cols = list(df.columns)


print('#'*50)

for i in range(len(cols)):

    print(cols[i], ': {:.3f}'.format(df[cols[i]].skew()), end='  >>>>  ')

    q1 = df[cols[i]].quantile(0.50)
    q2 = df[cols[i]].quantile(0.90)
    
    df[cols[i]] = np.where(df[cols[i]] > q2, q1, df[cols[i]])

    print('{:.3f}'.format(df[cols[i]].skew()))



# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1



# for i in range(len(cols)):
#     # print(cols[i], ': {:.3f}'.format(df[cols[i]].skew()), end='  >>>>  ')
#     q1 = df[cols[i]].quantile(0.10)
#     q2 = df[cols[i]].quantile(0.90)
    
#     df[cols[i]] = np.where(df[cols[i]] < q1, q1, df[cols[i]])
#     df[cols[i]] = np.where(df[cols[i]] > q2, q2, df[cols[i]])
#     # print('{:.3f}'.format(df[cols[i]].skew()))



# kmeans = KMeans()
# y_kmeans = kmeans.fit_predict(df)
# df['y_kmeans'] = y_kmeans
# print('kMeans value counts are:\n')
# print(df['y_kmeans'].value_counts())


