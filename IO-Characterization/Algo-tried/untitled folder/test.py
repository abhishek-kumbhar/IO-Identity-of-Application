
import pandas as pd
import statistics as st

s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

s1 = pd.Series([1, 1, 3, 4, 5, 6, 7, 8])
s2 = pd.Series([1, 1, 3, 4])
s3 = pd.Series([5, 6, 7, 8])


a = [11, 31, 21, 19, 8, 54, 35, 26, 23, 13, 29, 17]

a.sort()

print('a: ', a)

s4 = pd.Series(a)

q1 = s4.quantile(0.25)
q2 = s4.quantile(0.50)
q3 = s4.quantile(0.75)

iqr = q3 - q1

r1 = q1 - 1.5 * iqr
r2 = q3 + 1.5 * iqr

print('q1: ', q1)
print('q2: ', q2)
print('q3: ', q3)

print('iqr: ', iqr)

print('r1: ', r1)
print('r2: ', r2)


for i in a:
    if i > r1 and i < r2:
        print(i)


q1 = st.quantiles(a, method='inclusive')

print('stats: ', q1)


# print(s1)

# print(s1.describe())
# print(s1.median())
# print(s2.median())
# print(s3.median())

# print('**')

# print(s1.quantile(0.1))
# print(s1.quantile(0.2))
# print('0.25 : ', s1.quantile(0.25))

# print(s1.quantile(0.3))
# print(s1.quantile(0.4))
# print(s1.quantile(0.5))
# print('0.50 : ', s1.quantile(0.50))

# print(s1.quantile(0.6))
# print(s1.quantile(0.7))
# print('0.75 : ', s1.quantile(0.75))

# print(s1.quantile(0.8))
# print(s1.quantile(0.9))
# print(s1.quantile(1.0))





