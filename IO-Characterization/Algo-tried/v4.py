
import pandas as pd
import plotly.express as px
import plotly as pp
import itertools


new_df = pd.read_excel('/Users/abhi/Desktop/IO-Characterization/CSV-and-Excels/k_means_OP_finals_1.xlsx')

c = ['NPROCS', 'RUN_TIME', 'TOTAL_IO_TIME', 'TOTAL_POSIX_F_META_TIME', 'TOTAL_POSIX_F_READ_TIME', 'TOTAL_POSIX_F_WRITE_TIME', 'TOTAL_IO_PER_PROC', 'TOTAL_IO_OPS', 
        'TOTAL_MD_OPS', 'TOTAL_READ_TIME', 'TOTAL_WRITE_TIME', 'TOTAL_MD_TIME']


# print(new_df.head())

new_df['y_kmeans'] = new_df['y_kmeans'].map({0: 'Cluster_0', 1: 'Cluster_1', 2: 'Cluster_2'})

# print(new_df.head())



x = c
y = c

comb = list(itertools.product(x, y))

lst = []
for i in comb:
    if i[0] == i[1]:
        continue
    else:
        lst.append(i)

iCnt = 0
iLst = [0, 1, 5, 6, 11, 27, 77, 78, 83]
iLst = [0, 1, 77, 78]


new_df["y_kmeans"] = new_df["y_kmeans"].astype(str)

for i in lst:
    if iCnt in iLst:
        # print(i)
    # fig = px.scatter(new_df[i[0]], new_df[i[1]], color=new_df['cluster'], title=str(i[0])+'    V/S    '+str(i[1])).update_layout(xaxis_title=str(i[0]), yaxis_title=str(i[1]))
        fig = px.scatter(new_df[i[0]], new_df[i[1]], color=new_df['y_kmeans'], color_discrete_sequence=px.colors.qualitative.Set1, title=str(i[0])+'    V/S    '+str(i[1])).update_layout(xaxis_title=str(i[0]), yaxis_title=str(i[1]))
        f_name = str(str(iCnt) + '___' + i[0]) + '---' + str(i[1]) + '.png'
        # fig.write_image('OUT_KMEANS_1/' + f_name)
        fig.show()

    iCnt += 1


# for i in lst:

#     # fig = px.scatter(new_df[i[0]], new_df[i[1]], color=new_df['cluster'], title=str(i[0])+'    V/S    '+str(i[1])).update_layout(xaxis_title=str(i[0]), yaxis_title=str(i[1]))
#     fig = px.scatter(new_df[i[0]], new_df[i[1]], color=new_df['y_kmeans'], color_discrete_sequence=px.colors.qualitative.G10, title=str(i[0])+'    V/S    '+str(i[1])).update_layout(xaxis_title=str(i[0]), yaxis_title=str(i[1]))

#     f_name = str(str(iCnt) + '___' + i[0]) + '---' + str(i[1]) + '.png'
#     fig.write_image('OUT_KMEANS_1/' + f_name)

#     iCnt += 1
#     # fig.show()