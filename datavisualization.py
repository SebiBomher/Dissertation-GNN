from Scripts.datasetsClasses import LinearRegressionDataset
from Scripts.learn import LossFunction
import numpy as np
import pandas as pd
from Scripts.data_proccess import DataReader, DatasetSize, DatasetSizeNumber, Graph
import plotly.express as px
import os
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV

def BoxPlotSpeed(df: pd.DataFrame, path_save: str) -> None:
    df = df[df['Speed'].notna()]
    df['Day'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').weekday())
    fig = px.box(df, x='Day', y='Speed')

    fig.write_image(path_save)


def MapPlotSensors(df: pd.DataFrame, path_save: str, path_processed_data: str = None, datasize: DatasetSize = None) -> None:
    zoom = 8
    if datasize != None and path_processed_data != None:
        datanodes = Graph.get_nodes_ids_by_size(path_processed_data, datasize)
        df = df[df['ID'].isin(datanodes)]
        if datasize == DatasetSize.Experimental:
            zoom = 14
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="ID", hover_data=["Type", "Lanes"],
                            color_discrete_sequence=["black"], zoom=zoom, size_max=15, mapbox_style="open-street-map")

    fig.write_image(path_save)


def PieChartRoadwayType(df: pd.DataFrame, path_save: str) -> None:
    series = df['Type'].value_counts(ascending=False, dropna=True)
    dataframe = pd.DataFrame({'Type': series.index, 'count': series.values})
    dataframe = dataframe.replace({'CD': 'Coll/Dist', 'CH': 'Conventional Highway',
                                  'FF': 'Freeway-Freeway connector', 'FR': 'Off Ramp', 'HV': 'HOV', 'ML': 'Mainline', 'OR': 'On Ramp'})
    fig = px.pie(dataframe, values='count',
                 names='Type', title='Types of roads')

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
            
                fig.write_image("{0}_{1}_{2}_{3}.png".format(
                    path_save, epsilon, lamda, dataset.name))

def CorrelationSpeedFlow(df: pd.DataFrame, path_save: str, node_id : int) -> None:
    df = df[df['Speed'].notna()]
    df = df[df['Flow'].notna()]
    df = df[df["Station"] == node_id]
    df['Hour'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').hour)
    df = df.head(288)
    fig = px.scatter(df, x="Hour", y="Flow", color='Speed')

    fig.write_image(path_save)

def CorrelationSpeedOccupancy(df: pd.DataFrame, path_save: str, node_id : int) -> None:
    df = df[df['Speed'].notna()]
    df = df[df['Occupancy'].notna()]
    df = df[df["Station"] == node_id]
    df['Hour'] = df['Timestamp'].apply(
        lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').hour)
    df = df.head(288)
    fig = px.scatter(df, x="Hour", y="total_bill", color='Speed')

    fig.write_image(path_save)


def BarPlotSamplesObserved(df: pd.DataFrame, path_save: str) -> None:
    df = df.groupby(["Observed"]).size().reset_index(name='Count')
    fig = px.bar(df, x="Observed", y="Count")

    fig.write_image(path_save)

def TableFinalResultsDataset(dfSTCONV: pd.DataFrame, dfCUSTOM: pd.DataFrame, path_save: str, datasetsize : DatasetSize) -> None:
    df = pd.DataFrame(columns=["Type", "RMSE", "MAPE", "MAE", "MSE"])

    dfCUSTOM = dfCUSTOM[dfCUSTOM["Size"] == datasetsize.name]
    dfCUSTOM = dfCUSTOM[["Criterion","Loss"]]
    dfCUSTOM = dfCUSTOM.groupby(["Criterion"]).min().T

    dfSTCONV = dfSTCONV[dfCUSTOM["Size"] == datasetsize.name]
    dfSTCONV = dfSTCONV[["Criterion","Loss"]]
    dfSTCONV = dfSTCONV.groupby(["Criterion"]).min().T

    df = df.append({"Type" : "STCONV (minimum)","RMSE": format((float)(dfSTCONV["RMSE"].iloc[0]),'.2f'),"MAPE": format((float)(dfSTCONV["MAPE"].iloc[0]),'.2f'),"MAE": format((float)(dfSTCONV["MAE"].iloc[0]),'.2f'),"MSE": format((float)(dfSTCONV["MSE"].iloc[0]),'.2f')},ignore_index=True)
    df = df.append({"Type" : "Custom (minimum)","RMSE": format((float)(dfCUSTOM["RMSE"].iloc[0]),'.2f'),"MAPE": format((float)(dfCUSTOM["MAPE"].iloc[0]),'.2f'),"MAE": format((float)(dfCUSTOM["MAE"].iloc[0]),'.2f'),"MSE": format((float)(dfCUSTOM["MSE"].iloc[0]),'.2f')},ignore_index=True)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Type, df.RMSE, df.MAPE, df.MAE, df.MSE],
                    fill_color='lavender',
                    align='left'))
    ])


    fig.write_image(path_save)

def TableFinalResults(dfLR: pd.DataFrame, dfSTCONV: pd.DataFrame, dfCUSTOM: pd.DataFrame, path_save: str) -> None:
    df = pd.DataFrame(columns=["Type", "Size", "RMSE", "MAPE", "MAE", "MSE"])

    dfLR = dfLR[["Criterion","Loss"]]
    dfLR = dfLR.groupby(["Criterion"]).mean().T

    dfSTCONV = dfSTCONV[["Criterion","Loss","Size"]]
    dfCUSTOM = dfCUSTOM[["Criterion","Loss","Size"]]

    df = df.append({"Type" : "Linear Regression", "Size" : "All" ,"RMSE": format((float)(dfLR["RMSE"].iloc[0]),'.2f'),"MAPE": format((float)(dfLR["MAPE"].iloc[0]),'.2f'),"MAE": format((float)(dfLR["MAE"].iloc[0]),'.2f'),"MSE": format((float)(dfLR["MSE"].iloc[0]),'.2f')},ignore_index=True)
    
    
    for datasetsize in DatasetSize:
        dfSTCONVTemp = dfSTCONV[dfSTCONV["Size"] == datasetsize.name]
        dfSTCONVTemp = dfSTCONVTemp[["Criterion","Loss"]]
        dfSTCONVTemp = dfSTCONVTemp.groupby(["Criterion"]).min().T
        df = df.append({"Type" : "STCONV",
                         "Size" : datasetsize.name ,
                         "RMSE": format((float)(dfSTCONVTemp["RMSE"].iloc[0]),'.2f'),
                         "MAPE": format((float)(dfSTCONVTemp["MAPE"].iloc[0]),'.2f'),
                         "MAE": format((float)(dfSTCONVTemp["MAE"].iloc[0]),'.2f'),
                         "MSE": format((float)(dfSTCONVTemp["MSE"].iloc[0]),'.2f')},ignore_index=True)
    
    for datasetsize in DatasetSize:
        dfCUSTOMTemp = dfCUSTOM[dfCUSTOM["Size"] == datasetsize.name]
        dfCUSTOMTemp = dfCUSTOMTemp[["Criterion","Loss"]]
        dfCUSTOMTemp = dfCUSTOMTemp.groupby(["Criterion"]).min().T
        df = df.append({"Type" : "Custom",
                        "Size" : datasetsize.name ,
                        "RMSE": format((float)(dfCUSTOMTemp["RMSE"].iloc[0]),'.2f'),
                        "MAPE": format((float)(dfCUSTOMTemp["MAPE"].iloc[0]),'.2f'),
                        "MAE": format((float)(dfCUSTOMTemp["MAE"].iloc[0]),'.2f'),
                        "MSE": format((float)(dfCUSTOMTemp["MSE"].iloc[0]),'.2f')},ignore_index=True)
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[df.Type, df.Size, df.RMSE, df.MAPE, df.MAE, df.MSE],
                   fill_color='lavender',
                   align='left'))
    ])

    fig.write_image(path_save)

def BoxPlotResultsLR(dfLR: pd.DataFrame, path_save: str) -> None:
    dfLR = dfLR[dfLR["Loss"] < 100]
    for criterion in LossFunction.Criterions():
        dfTemp = dfLR[dfLR["Criterion"] == criterion.__name__]
        fig = px.box(dfTemp, x="Criterion", y="Loss")
        fig.write_image(os.path.join(path_save,"boxplot_result_LR_{0}.png".format(criterion.__name__)))

def BoxPlotResults(dfSTCONV: pd.DataFrame, dfCUSTOM: pd.DataFrame, path_save: str) -> None:

    dfSTCONV["Type"] = "STCONV"
    dfCUSTOM["Type"] = "Custom"
    df = dfSTCONV.append(dfCUSTOM,ignore_index=True)
    for criterion in LossFunction.Criterions():
        dfTemp = df[df["Criterion"] == criterion.__name__]
        fig = px.box(dfTemp, x="Size", y="Loss", color="Type")
        fig.write_image(os.path.join(path_save,"boxplot_result_{0}.png".format(criterion.__name__)))

def RegressionLRTruePredicted(dfLR : pd.DataFrame, datareader : DataReader,proccessed_data_path : str,path_save : str) -> None:
    #TODO
    
    device = "cpu"
    parameters = {
        'normalize':[True],
    }

    lr_model = LinearRegression()
    clf = GridSearchCV(lr_model, parameters, refit=True, cv=5)
    for index, (X_train, X_test, Y_train, Y_test, node_id) in enumerate(LinearRegressionDataset(proccessed_data_path,datareader,device)):
        best_model = clf.fit(X_train,Y_train)
        y_pred = best_model.predict(X_test)
        Y_test = [item for sublist in Y_test.tolist() for item in sublist]
        y_pred = [item for sublist in y_pred.tolist() for item in sublist]
        dict1 = dict(Time = np.arange(len(y_pred)),Actual = Y_test,Predicted = y_pred)
        df = pd.DataFrame(dict1)
        fig = px.line(df, x='Time', y=["Actual","Predicted"], title="Prediction for node {0}".format(node_id))
        fig.write_image(os.path.join(path_save,"RegressionLRTruePredicted","{0}.png".format(node_id)))
        print("Regression LR True Predicted node {0}".format(node_id))
        break

def RegressionLoss(dfSTCONV: pd.DataFrame, dfCUSTOM: pd.DataFrame,path_save : str) -> None:
    #TODO 
    dfPlot = pd.DataFrame()
    data = {}
    dfPlot["Time"] = np.arange(300)
    for size in DatasetSize:
        if size != DatasetSize.Medium:
            dfSTCONVTemp = dfSTCONV[dfSTCONV["Criterion"] == "MAE"]
            dfSTCONVTemp = dfSTCONVTemp[dfSTCONVTemp["Size"] == size.name]
            dfSTCONVTemp2 = dfSTCONVTemp[["Criterion","Loss"]]
            dfSTCONVTemp2 = dfSTCONVTemp2.groupby(["Criterion"]).min()
            minLoss = dfSTCONVTemp2["Loss"].iloc[0]
            BEstTrial = dfSTCONVTemp[dfSTCONVTemp["Loss"] == minLoss]
            df = dfSTCONVTemp[dfSTCONVTemp["Trial"] == BEstTrial["Trial"].iloc[0]]
            datalist = df["Loss"].tolist()
            datalist.extend(np.zeros(300 - len(datalist)))
            dfPlot["STCONV_{0}".format(size.name)] = datalist

        dfCUSTOMTemp = dfCUSTOM[dfCUSTOM["Criterion"] == "MAE"]
        dfCUSTOMTemp = dfCUSTOMTemp[dfCUSTOMTemp["Size"] == size.name]
        dfCUSTOMTemp2 = dfCUSTOMTemp[["Criterion","Loss"]]
        dfCUSTOMTemp2 = dfCUSTOMTemp2.groupby(["Criterion"]).min()
        minLoss = dfCUSTOMTemp2["Loss"].iloc[0]
        BEstTrial = dfCUSTOMTemp[dfCUSTOMTemp["Loss"] == minLoss]
        df = dfCUSTOMTemp[dfCUSTOMTemp["Trial"] == BEstTrial["Trial"].iloc[0]]
        datalist = df["Loss"].tolist()
        datalist.extend(np.zeros(300 - len(datalist)))
        dfPlot["CUSTOM_{0}".format(size.name)] = datalist
    columns = dfPlot.columns.tolist()
    columns.pop(0)
    # print(columns)
    fig = px.line(dfPlot, x="Time", y= columns)
    fig.show()
    fig.write_image(os.path.join(path_save,"Training_no_medium.png"))

def HeatMapLoss(dfInfo : pd.DataFrame,dfMeta : pd.DataFrame,path_save : str) -> None:
    dfInfo = dfInfo[dfInfo["Criterion"] == "MAE"]
    dfInfo = dfInfo[dfInfo["Loss"] < 20]
    df = pd.merge(dfMeta, dfInfo, left_on='ID', right_on='Node_Id')
    fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Loss",
                            color_continuous_scale=px.colors.sequential.Bluered, zoom=8, mapbox_style="open-street-map", title='Map for Loss MAE')

    fig.write_image(path_save)

def BarPlotWaitingTimes(df : pd.DataFrame,path_save : str) -> None:
    #TODO
    fig = px.box(df, x="Size", y="Loss", color="Type")
    fig.write_image(path_save)

if __name__ == '__main__':
    path_data = "E:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    path_save_plots = "E:\\FacultateMasterAI\\Dissertation-GNN\\Plots"
    path_processed_data = "E:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
    path_results= "E:\\FacultateMasterAI\\Dissertation-GNN\\Results"
    if not os.path.exists(path_save_plots):
        os.mkdir(path_save_plots)
    datareader = DataReader(path_data, graph_info_txt)
    # dfInfo, dfMeta = datareader.visualization()
    # BoxPlotSpeed(dfInfo,os.path.join(path_save_plots,"boxplot.png"))
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotAll.png"))
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotExperimental.png"),path_processed_data,DatasetSize.Experimental)
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotSmall.png"),path_processed_data,DatasetSize.Small)
    # MapPlotSensors(dfMeta,os.path.join(path_save_plots,"mapplotMedium.png"),path_processed_data,DatasetSize.Medium)
    # PieChartRoadwayType(dfMeta,os.path.join(path_save_plots,"piechart.png"))
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat9.png"),9)
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat15.png"),15)
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat18.png"),18)
    # MapHeatmapSpeed(dfInfo,dfMeta,os.path.join(path_save_plots,"mapheat22.png"),22)
    # GraphHeatmap(path_processed_data, os.path.join(path_save_plots, "graph.png"), datareader)
    # BarPlotSamplesObserved(dfInfo,os.path.join(path_save_plots,"barplot.png"))

    dfLR,dfSTCONV,dfCUSTOM = datareader.results(path_results)
    # TableFinalResults(dfLR,dfSTCONV,dfCUSTOM,os.path.join(path_save_plots,"tableresults.png"))
    # BoxPlotResults(dfSTCONV,dfCUSTOM,path_save_plots)
    # BoxPlotResultsLR(dfLR,path_save_plots)
    # RegressionLRTruePredicted(dfLR,datareader,path_processed_data,path_save_plots)
    # HeatMapLoss(dfLR,dfMeta,os.path.join(path_save_plots,"HeatMapLRLossMAE.png"))
    RegressionLoss(dfSTCONV,dfCUSTOM,path_save_plots)