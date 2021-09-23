import numpy as np
import pandas as pd
from Scripts.data_proccess import DataReader, DatasetSize, DatasetSizeNumber, Graph
import plotly.express as px
import os
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def BoxPlotSpeed(df : pd.DataFrame, path_save : str) -> None:
    df = df[df['Speed'].notna()]
    df['Day'] = df['Timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %H:%M:%S').weekday())
    fig = px.box(df, x='Day', y='Speed')
    fig.show()
    fig.write_image(path_save)

def MapPlotSensors(df : pd.DataFrame, path_save : str, path_processed_data : str = None, datasize : DatasetSize = None) -> None:
    zoom = 8
    if datasize != None and path_processed_data != None:
        datanodes = Graph.get_nodes_ids_by_size(path_processed_data,datasize)
        df = df[df['ID'].isin(datanodes)]
        if datasize == DatasetSize.Experimental:
            zoom = 15
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="ID", hover_data=["Type", "Lanes"],
    color_discrete_sequence=["black"], zoom=zoom, size_max=15,mapbox_style="open-street-map")
    fig.show()
    fig.write_image(path_save)

def PieChartRoadwayType(df : pd.DataFrame, path_save : str) -> None:
    series = df['Type'].value_counts(ascending=False,dropna=True)
    dataframe = pd.DataFrame({'Type':series.index, 'count':series.values})
    dataframe = dataframe.replace({'CD': 'Coll/Dist', 'CH': 'Conventional Highway', 'FF': 'Freeway-Freeway connector', 'FR': 'Off Ramp', 'HV': 'HOV', 'ML' : 'Mainline','OR' : 'On Ramp'})
    fig = px.pie(dataframe, values='count', names='Type', title='Types of roads')
    fig.show()
    fig.write_image(path_save)

def MapHeatmapSpeed(dfInfo : pd.DataFrame,dfMeta : pd.DataFrame, path_save : str, hour : int) -> None:
    dfInfo2 = dfInfo[dfInfo['Speed'].notna()]
    dfInfo2['Hour'] = dfInfo2['Timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %H:%M:%S').hour)
    dfInfo2 = dfInfo2.loc[dfInfo2['Hour'] == hour]
    dfInfo2 = dfInfo2[['Hour','Station','Speed']]
    dfInfo2 = dfInfo2.groupby(['Hour','Station']).mean()
    df = pd.merge(dfMeta,dfInfo2,left_on='ID',right_on='Station')
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Speed",
    color_continuous_scale=px.colors.sequential.Bluered, zoom=8, mapbox_style="open-street-map",title='Traffic speed at {0}:00'.format(str(hour)))
    fig.show()
    fig.write_image(path_save)

def GraphHeatmap(proccessed_data_path : str, path_save : str, datareader : DataReader) -> None:
    epsilon_array = [0.1, 0.3, 0.5, 0.7]
    lamda_array = [1, 3, 5, 10]
    for epsilon in epsilon_array:
        for lamda in lamda_array:
            graph = Graph(proccessed_data_path,epsilon,lamda,DatasetSize.Medium,datareader)
            graph_matrix = np.zeros((DatasetSizeNumber.Medium.value, DatasetSizeNumber.Medium.value))
            for index,(edge_index,edge_weight) in enumerate(zip(np.transpose(graph.edge_index),graph.edge_weight)):
                graph_matrix[edge_index[0]][edge_index[1]] = edge_weight
                graph_matrix[edge_index[1]][edge_index[0]] = edge_weight
            fig = px.imshow(graph_matrix,title='Graph Heatmap for epsilon {0} and sigma {1}'.format(epsilon,lamda))
            fig.show()
            fig.write_image("{0}_{1}_{2}".format(path_save,epsilon,lamda))

def BarPlotSamplesObserved(df : pd.DataFrame, path_save : str) -> None:
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    fig.write_image(path_save)

def RegresionLossFunciton(df : pd.DataFrame, path_save : str) -> None:
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    fig.write_image(path_save)

def TruePredictedRegression(df : pd.DataFrame, path_save : str) -> None:
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    fig.write_image(path_save)

def TableFinalResults(df : pd.DataFrame, path_save : str) -> None:
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    fig.write_image(path_save)


if __name__ == '__main__':
    # TODO
    path_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    path_save_plots = "D:\\FacultateMasterAI\\Dissertation-GNN\\Plots"
    path_processed_data = "D:\\FacultateMasterAI\\Proccessed"
    if not os.path.exists(path_save_plots):
        os.mkdir(path_save_plots)
    datareader = DataReader(path_data,graph_info_txt)
    dfInfo,dfMeta = datareader.visualization()
    # BoxPlotSpeed(dfInfo,os.path.join(path_save_plots,"boxplot.png"))
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotAll.png"))
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotExperimental.png"),path_processed_data,DatasetSize.Experimental)
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotSmall.png"),path_processed_data,DatasetSize.Small)
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotMedium.png"),path_processed_data,DatasetSize.Medium)
    # PieChartRoadwayType(dfMeta,os.path.join(path_save_plots,"piechart.png"))
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),9)
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),15)
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),18)
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),22)
    GraphHeatmap(path_processed_data,os.path.join(path_save_plots,"graph.png"),datareader)
    # BarPlotSamplesObserved(dfInfo,os.path.join(path_save_plots,"barplot.png"))
    # TableFinalResults(dfMeta,os.path.join(path_save_plots,"tablefinal.png"))