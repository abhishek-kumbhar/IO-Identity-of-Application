
import pandas as pd
import plotly.express as px
import plotly as pp
import itertools

import numpy as np

import matplotlib.pyplot as plt



new_df = pd.read_excel('/Users/abhi/Desktop/IO-Characterization/CSV-and-Excels/9_clusters.xlsx')

c = ['NPROCS', 'RUN_TIME', 'TOTAL_IO_TIME', 'TOTAL_POSIX_F_META_TIME', 'TOTAL_POSIX_F_READ_TIME', 'TOTAL_POSIX_F_WRITE_TIME', 'TOTAL_IO_PER_PROC', 'TOTAL_IO_OPS', 
        'TOTAL_MD_OPS', 'TOTAL_READ_TIME', 'TOTAL_WRITE_TIME', 'TOTAL_MD_TIME']


new_df['cluster_colours'] = new_df['cluster_colours'].map({0: 'Cluster_0', 1: 'Cluster_1', 2: 'Cluster_2', 3: 'Cluster_3', 4: 'Cluster_4', 5: 'Cluster_5', 6: 'Cluster_6', 7: 'Cluster_7', 8: 'Cluster_8', 9: 'Cluster_9'})


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

new_df["cluster_colours"] = new_df["cluster_colours"].astype(str)

# print(new_df.columns)

print(new_df['NPROCS'].describe())

l = [x for x in range(2050)]


layout = dict(
    xaxis=dict(
        tickmode="array",
        tickvals=l,
        ticktext=['0'],
        # tickformat='%Y-%m-%d',
        tickangle=0,
        title='Hello',
        showgrid=True
    )
)



fig = px.scatter(new_df['NPROCS'], new_df['TOTAL_IO_TIME'], color=new_df['cluster_colours'], color_discrete_sequence=px.colors.qualitative.G10, title=str(i[0])+'    V/S    '+str(i[1]))

fig.update_layout(layout)
fig.show()



# for i in lst[:]:
#     if iCnt in iLst:

#     # fig = px.scatter(new_df[i[0]], new_df[i[1]], color=new_df['cluster'], title=str(i[0])+'    V/S    '+str(i[1])).update_layout(xaxis_title=str(i[0]), yaxis_title=str(i[1]))
#         fig = px.scatter(new_df[i[0]], new_df[i[1]], color=new_df['cluster_colours'], color_discrete_sequence=px.colors.qualitative.G10, title=str(i[0])+'    V/S    '+str(i[1])).update_layout(xaxis_title=str(i[0]), yaxis_title=str(i[1]))
        
#         # fig.update_xaxes(tickmode = 'array', tickvals = [x for x in range(9)], ticktext=[str(x) for x in range(9)])
    
#         f_name = str(str(iCnt) + '___' + i[0]) + '---' + str(i[1]) + '.png'
#         # fig.write_image('OUT/' + f_name)
#         fig.show()

#     iCnt += 1


