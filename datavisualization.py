from Scripts.learn import LossFunction
import numpy as np
import pandas as pd
from Scripts.data_proccess import DataReader, DatasetSize, DatasetSizeNumber, Graph
import plotly.express as px
import os
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def BoxPlotSpeed(df: pd.DataFrame, path_save: str) -> None:
    df = df[df['Speed'].notna()]
    df['Day'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').weekday())
    fig = px.box(df, x='Day', y='Speed')
    fig.show()
    fig.write_image(path_save)


def MapPlotSensors(df: pd.DataFrame, path_save: str, path_processed_data: str = None, datasize: DatasetSize = None) -> None:
    zoom = 8
    if datasize != None and path_processed_data != None:
        datanodes = Graph.get_nodes_ids_by_size(path_processed_data, datasize)
        df = df[df['ID'].isin(datanodes)]
        if datasize == DatasetSize.Experimental:
            zoom = 15
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="ID", hover_data=["Type", "Lanes"],
                            color_discrete_sequence=["black"], zoom=zoom, size_max=15, mapbox_style="open-street-map")
    fig.show()
    fig.write_image(path_save)


def PieChartRoadwayType(df: pd.DataFrame, path_save: str) -> None:
    series = df['Type'].value_counts(ascending=False, dropna=True)
    dataframe = pd.DataFrame({'Type': series.index, 'count': series.values})
    dataframe = dataframe.replace({'CD': 'Coll/Dist', 'CH': 'Conventional Highway',
                                  'FF': 'Freeway-Freeway connector', 'FR': 'Off Ramp', 'HV': 'HOV', 'ML': 'Mainline', 'OR': 'On Ramp'})
    fig = px.pie(dataframe, values='count',
                 names='Type', title='Types of roads')
    fig.show()
    fig.write_image(path_save)


def MapHeatmapSpeed(dfInfo: pd.DataFrame, dfMeta: pd.DataFrame, path_save: str, hour: int) -> None:
    dfInfo2 = dfInfo[dfInfo['Speed'].notna()]
    dfInfo2['Hour'] = dfInfo2['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').hour)
    dfInfo2 = dfInfo2.loc[dfInfo2['Hour'] == hour]
    dfInfo2 = dfInfo2[['Hour', 'Station', 'Speed']]
    dfInfo2 = dfInfo2.groupby(['Hour', 'Station']).mean()
    df = pd.merge(dfMeta, dfInfo2, left_on='ID', right_on='Station')
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Speed",
                            color_continuous_scale=px.colors.sequential.Bluered, zoom=8, mapbox_style="open-street-map", title='Traffic speed at {0}:00'.format(str(hour)))
    fig.show()
    fig.write_image(path_save)


def GraphHeatmap(proccessed_data_path: str, path_save: str, datareader: DataReader) -> None:
    epsilon_array = [0.1, 0.3, 0.5, 0.7]
    lamda_array = [1, 3, 5, 10]
    for epsilon in epsilon_array:
        for lamda in lamda_array:
            for dataset in DatasetSize:
                graph = Graph(proccessed_data_path, epsilon,
                              lamda, dataset, datareader)
                nr_nodes = Graph.get_number_nodes_by_size(dataset)
                graph_matrix = np.zeros((nr_nodes, nr_nodes))
                for index, (edge_index, edge_weight) in enumerate(zip(np.transpose(graph.edge_index), graph.edge_weight)):
                    graph_matrix[edge_index[0]][edge_index[1]] = edge_weight
                    graph_matrix[edge_index[1]][edge_index[0]] = edge_weight
                fig = px.imshow(graph_matrix, title='Graph Heatmap for epsilon {0} and sigma {1} with size {2}'.format(
                    epsilon, lamda, dataset.name))
                fig.show()
                fig.write_image("{0}_{1}_{2}_{3}.png".format(
                    path_save, epsilon, lamda, dataset.name))

def Training(dfResult: pd.DataFrame, path_save: str) -> None:
    dfResult = dfResult.groupby(["Trial","Size"])
    fig = px.line(dfResult, x="Trial", y="Loss", color='Size')
    fig.show()
    fig.write_image(path_save)

def CorrelationSpeedFlow(df: pd.DataFrame, path_save: str, node_id : int) -> None:
    df = df[df['Speed'].notna()]
    df = df[df['Flow'].notna()]
    df = df[df["Station"] == node_id]
    df['Hour'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').hour)
    df = df.head(288)
    fig = px.scatter(df, x="Hour", y="Flow", color='Speed')
    fig.show()
    fig.write_image(path_save)

def CorrelationSpeedOccupancy(df: pd.DataFrame, path_save: str, node_id : int) -> None:
    df = df[df['Speed'].notna()]
    df = df[df['Occupancy'].notna()]
    df = df[df["Station"] == node_id]
    df['Hour'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').hour)
    df = df.head(288)
    fig = px.scatter(df, x="Hour", y="total_bill", color='Speed')
    fig.show()
    fig.write_image(path_save)


def BarPlotSamplesObserved(df: pd.DataFrame, path_save: str) -> None:
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    fig.write_image(path_save)


def BoxPlotResults(df: pd.DataFrame, path_save: str) -> None:
    df = df[["Criterion","Loss"]]
    fig = px.box(df, x="Criterion", y="Loss")
    fig.show()
    fig.write_image(path_save)


def MapPlotResults(dfInfo: pd.DataFrame, dfMeta: pd.DataFrame, path_save: str) -> None:
    for criterion in LossFunction.Criterions:
        dfInfo = dfInfo[dfInfo["Criterion"] == criterion.__name__]
    dfInfo = dfInfo.groupby(["Criterion","Node_Id"]).min()
    df = pd.merge(dfMeta, dfInfo, left_on='ID', right_on='Node_ID')
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Speed",
                            color_continuous_scale=px.colors.sequential.Bluered, zoom=8, mapbox_style="open-street-map", title='Result Map for Function {0}'.format(criterion.__name__))
    fig.show()
    fig.write_image(path_save)


def TableFinalResults(dfLR: pd.DataFrame, dfSTCONV: pd.DataFrame, dfCUSTOM: pd.DataFrame, path_save: str) -> None:
    df = pd.DataFrame(columns=["Type", "RMSE", "MAPE", "MAE", "MSE"])

    dfLR = dfLR[["Criterion","Loss"]]
    dfLR = dfLR.groupby(["Criterion"]).min()

    dfSTCONV = dfSTCONV[["Criterion","Loss"]]
    dfSTCONV = dfSTCONV.groupby(["Criterion"]).min()

    dfCUSTOM = dfCUSTOM[["Criterion","Loss"]]
    dfCUSTOM = dfCUSTOM.groupby(["Criterion"]).min()

    df.append({"Type" : "Linear Regression","RMSE": dfLR["RMSE"],"MAPE": dfLR["MAPE"],"MAE": dfLR["MAE"],"MSE": dfLR["MSE"]})
    df.append({"Type" : "STCONV","RMSE": dfSTCONV["RMSE"],"MAPE": dfSTCONV["MAPE"],"MAE": dfSTCONV["MAE"],"MSE": dfSTCONV["MSE"]})
    df.append({"Type" : "Custom","RMSE": dfCUSTOM["RMSE"],"MAPE": dfCUSTOM["MAPE"],"MAE": dfCUSTOM["MAE"],"MSE": dfCUSTOM["MSE"]})

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Type, df.RMSE, df.MAPE, df.MAE, df.MSE],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.show()
    fig.write_image(path_save)


if __name__ == '__main__':
    # TODO
    path_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    path_save_plots = "D:\\FacultateMasterAI\\Dissertation-GNN\\Plots"
    path_processed_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
    path_results= "D:\\FacultateMasterAI\\Dissertation-GNN\\Results"
    if not os.path.exists(path_save_plots):
        os.mkdir(path_save_plots)
    datareader = DataReader(path_data, graph_info_txt)
    dfInfo, dfMeta = datareader.visualization()
    dfLR,dfSTCONV,dfCUSTOM = datareader.results(path_results)
    BoxPlotSpeed(dfInfo,os.path.join(path_save_plots,"boxplot.png"))
    MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotAll.png"))
    MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotExperimental.png"),path_processed_data,DatasetSize.Experimental)
    MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotSmall.png"),path_processed_data,DatasetSize.Small)
    MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotMedium.png"),path_processed_data,DatasetSize.Medium)
    PieChartRoadwayType(dfMeta,os.path.join(path_save_plots,"piechart.png"))
    MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),9)
    MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),15)
    MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),18)
    MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat.png"),22)
    GraphHeatmap(path_processed_data, os.path.join(path_save_plots, "graph.png"), datareader)
    BoxPlotResults(dfLR,path_processed_data, os.path.join(path_save_plots, "BoxPlotLR.png"), datareader)
    BoxPlotResults(dfSTCONV,path_processed_data, os.path.join(path_save_plots, "BoxPlotSTCONV.png"), datareader)
    BoxPlotResults(dfCUSTOM,path_processed_data, os.path.join(path_save_plots, "BoxPlotCUSTOM.png"), datareader)
    MapPlotResults(dfLR, os.path.join(path_save_plots, "MapPlotResults.png"), datareader)
    TableFinalResults(dfLR,dfSTCONV,dfCUSTOM,os.path.join(path_save_plots,"tablefinal.png"))
    Training(dfSTCONV,os.path.join(path_save_plots,"trainingSTCONV.png"))
    Training(dfCUSTOM,os.path.join(path_save_plots,"trainingCustom.png"))
    # BarPlotSamplesObserved(dfInfo,os.path.join(path_save_plots,"barplot.png"))
