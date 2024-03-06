
import pandas as pd
import plotly.express as px

new_df = pd.read_excel('/Users/abhi/Desktop/IO-Characterization/CSV-and-Excels/KMeans_Data.xlsx')

new_df['y_kmeans'] = new_df['y_kmeans'].map({0: 'Cluster_0', 1: 'Cluster_1', 2: 'Cluster_2'})
new_df['y_kmeans'] = new_df['y_kmeans'].astype(str)


lst = [('NPROCS', 'RUN_TIME'), ('NPROCS', 'TOTAL_IO_TIME')]

for i in range(len(lst)):
    fig = px.scatter(new_df[lst[i][0]], new_df[lst[i][1]], color=new_df['y_kmeans'], color_discrete_sequence=px.colors.qualitative.Dark24, title=str(lst[i][0])+'    V/S    '+str(lst[i][1])).update_layout(xaxis_title=str(lst[i][0]), yaxis_title=str(lst[i][1]))
    fig.show()




new_df = pd.read_excel('/Users/abhi/Desktop/IO-Characterization/CSV-and-Excels/DBSCAN_Data.xlsx')

new_df['cluster_colours'] = new_df['cluster_colours'].map({0: 'Cluster_0', 1: 'Cluster_1', 2: 'Cluster_2', 3: 'Cluster_3', 4: 'Cluster_4', 5: 'Cluster_5', 6: 'Cluster_6', 7: 'Cluster_7', 8: 'Cluster_8', 9: 'Cluster_9'})
new_df['cluster_colours'] = new_df['cluster_colours'].astype(str)


lst = [('NPROCS', 'RUN_TIME'), ('NPROCS', 'TOTAL_IO_TIME'), ('TOTAL_IO_OPS', 'NPROCS'), ('TOTAL_IO_OPS', 'RUN_TIME')]

for i in range(len(lst)):
    fig = px.scatter(new_df[lst[i][0]], new_df[lst[i][1]], color=new_df['cluster_colours'], color_discrete_sequence=px.colors.qualitative.Dark24, title=str(lst[i][0])+'    V/S    '+str(lst[i][1])).update_layout(xaxis_title=str(lst[i][0]), yaxis_title=str(lst[i][1]))
    fig.show()