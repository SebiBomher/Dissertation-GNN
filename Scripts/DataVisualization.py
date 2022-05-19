#region Imports

from black import Mode
import numpy as np
import pandas as pd
import plotly.express as px
import os
import datetime
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from Scripts.DatasetClasses import LinearRegressionDataset
from Scripts.Learn import LossFunction
from Scripts.Utility import DistanceType, Folders, ModelType, Constants
from Scripts.DataProccess import DataReader, DatasetSize, Graph

#endregion


class DataViz():
    r"""
        Class used to create different plots for Data Visualization. 
        It covers dataset data visualization and results data visualization
    """

    def __init__(self, path_data: str, graph_info_txt: str,
                 path_save_plots: str, path_processed_data: str,
                 path_results: str, dfInfo: pd.DataFrame, dfMeta: pd.DataFrame,
                 dfLR: pd.DataFrame, dfARIMA: pd.DataFrame,
                 dfSARIMA: pd.DataFrame, dfSTCONV: pd.DataFrame,
                 dfLSTM: pd.DataFrame, dfDCRNN: pd.DataFrame):

        self.path_data = path_data
        self.graph_info_txt = graph_info_txt
        self.path_save_plots = path_save_plots
        self.path_processed_data = path_processed_data
        self.path_results = path_results
        self.dfInfo = dfInfo
        self.dfMeta = dfMeta
        self.dfLR = dfLR
        self.dfARIMA = dfARIMA
        self.dfSARIMA = dfSARIMA
        self.dfSTCONV = dfSTCONV
        self.dfLSTM = dfLSTM
        self.dfDCRNN = dfDCRNN

    def BoxPlotSpeed(self, name_save: str) -> None:
        r"""
            Plot for Box Plot over different days for speed
        """
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        df = self.dfInfo
        df = df[df['Speed'].notna()]
        df['Day'] = df['Timestamp'].apply(lambda x: datetime.datetime.strptime(
            x, '%m/%d/%Y %H:%M:%S').weekday())
        fig = px.box(df, x='Day', y='Speed')

        fig.write_image(path_save)

    def MapPlotSensors(self, name_save: str, datasize: DatasetSize) -> None:
        r"""
            Plot for sensors on Los Angeles Map
        """
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        df = self.dfMeta
        zoom = 8
        datanodes = Graph.get_nodes_ids_by_size(datasize)
        df = df[df['ID'].isin(datanodes)]
        if datasize == DatasetSize.Experimental:
            zoom = 14
        if datasize == DatasetSize.Tiny:
            zoom = 12
        fig = px.scatter_mapbox(df,
                                lat="Latitude",
                                lon="Longitude",
                                hover_name="ID",
                                hover_data=["Type", "Lanes"],
                                color_discrete_sequence=["black"],
                                zoom=zoom,
                                size_max=15,
                                mapbox_style="open-street-map")

        fig.write_image(path_save)

    def PieChartRoadwayType(self, name_save: str) -> None:
        r"""
            Pie chart with the different road types
        """
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        df = self.dfMeta
        series = df['Type'].value_counts(ascending=False, dropna=True)
        dataframe = pd.DataFrame({
            'Type': series.index,
            'count': series.values
        })
        dataframe = dataframe.replace({
            'CD': 'Coll/Dist',
            'CH': 'Conventional Highway',
            'FF': 'Freeway-Freeway connector',
            'FR': 'Off Ramp',
            'HV': 'HOV',
            'ML': 'Mainline',
            'OR': 'On Ramp'
        })
        fig = px.pie(dataframe,
                     values='count',
                     names='Type',
                     title='Types of roads')

        fig.write_image(path_save)

    def MapHeatmapSpeed(self, name_save: str, hour: int) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        dfInfo = self.dfInfo
        dfMeta = self.dfMeta

        dfInfo2 = dfInfo[dfInfo['Speed'].notna()]
        dfInfo2['Hour'] = dfInfo2['Timestamp'].apply(
            lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').hour)
        dfInfo2 = dfInfo2.loc[dfInfo2['Hour'] == hour]
        dfInfo2 = dfInfo2[['Hour', 'Station', 'Speed']]
        dfInfo2 = dfInfo2.groupby(['Hour', 'Station']).mean()
        df = pd.merge(dfMeta, dfInfo2, left_on='ID', right_on='Station')
        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Speed",
            color_continuous_scale=px.colors.sequential.Bluered,
            zoom=8,
            mapbox_style="open-street-map",
            title='Traffic speed at {0}:00'.format(str(hour)))

        fig.write_image(path_save)

    def GetGraphMatrixFromGraph(graph: Graph, size: DatasetSize):
        nr_nodes = Graph.get_number_nodes_by_size(size)
        graph_matrix = np.zeros((nr_nodes, nr_nodes))
        for index, (edge_index, edge_weight) in enumerate(
                zip(np.transpose(graph.edge_index), graph.edge_weight)):
            graph_matrix[edge_index[0]][edge_index[1]] = edge_weight
            # graph_matrix[edge_index[1]
            #             ][edge_index[0]] = edge_weight
        return graph_matrix

    def GraphHeatmap(self, name_save: str, datareader: DataReader) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        epsilon_array = [0.1, 0.3, 0.5, 0.7]
        sigma_array = [1, 3, 5, 10]
        for epsilon in epsilon_array:
            for sigma in sigma_array:
                for distanceType in DistanceType:
                    for dataset in DatasetSize:
                        if (dataset != DatasetSize.ExperimentalManual
                                and dataset != DatasetSize.ExperimentalLR
                                and dataset != DatasetSize.TinyManual
                                and dataset != DatasetSize.TinyLR
                                and dataset != DatasetSize.All):
                            graph = Graph(epsilon, sigma, dataset, datareader)
                            graph_matrix = DataViz.GetGraphMatrixFromGraph(
                                graph, dataset)
                            fig = px.imshow(
                                graph_matrix,
                                title='Graph Heatmap for epsilon {0} and sigma {1} with size {2}'
                                .format(epsilon, sigma, dataset.name))

                            fig.write_image("{0}_{1}_{2}_{3}.png".format(
                                path_save, epsilon, sigma, dataset.name))
        for dataset in [
                DatasetSize.ExperimentalManual, DatasetSize.ExperimentalLR,
                DatasetSize.TinyManual, DatasetSize.TinyLR
        ]:
            graph = Graph(0.1, 1, dataset, datareader)
            graph_matrix = DataViz.GetGraphMatrixFromGraph(graph, dataset)
            fig = px.imshow(graph_matrix,
                            title='Graph Heatmap for size {0}'.format(
                                dataset.name))
            fig.write_image("{0}.png".format(dataset.name))

    def TableFinalResults(self, name_save: str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        dfLR = self.dfLR
        dfARIMA = self.dfARIMA
        dfSARIMA = self.dfSARIMA
        dfSTCONV = self.dfSTCONV
        dfLSTM = self.dfLSTM
        dfDCRNN = self.dfDCRNN

        dfResults = dfSTCONV.append(dfDCRNN, ignore_index=True)
        dfResults = dfResults.append(dfLSTM, ignore_index=True)

        df = pd.DataFrame(
            columns=["Model", "Size", "Distance", "Generation", "RMSE", "MAPE", "MAE", "MSE"])

        dfLR = dfLR[["Criterion", "Loss"]]
        dfLR = dfLR.groupby(["Criterion"]).mean().T

        dfARIMA = dfARIMA[["Criterion", "Loss"]]
        dfARIMA = dfARIMA.groupby(["Criterion"]).mean().T

        dfSARIMA = dfSARIMA[["Criterion", "Loss"]]
        dfSARIMA = dfSARIMA.groupby(["Criterion"]).mean().T

        df = df.append(
            {
                "Model": "Linear Regression",
                "Size": "All",
                "Distance": "-",
                "Generation": "-",
                "RMSE": format((float)(dfLR["RMSE"].iloc[0]), '.2f'),
                "MAPE": format((float)(dfLR["MAPE"].iloc[0]), '.2f'),
                "MAE": format((float)(dfLR["MAE"].iloc[0]), '.2f'),
                "MSE": format((float)(dfLR["MSE"].iloc[0]), '.2f')
            },
            ignore_index=True)

        df = df.append(
            {
                "Model": "ARIMA",
                "Size": "All",
                "Distance": "-",
                "Generation": "-",
                "RMSE": format((float)(dfARIMA["RMSE"].iloc[0]), '.2f'),
                "MAPE": format((float)(dfARIMA["MAPE"].iloc[0]), '.2f'),
                "MAE": format((float)(dfARIMA["MAE"].iloc[0]), '.2f'),
                "MSE": format((float)(dfARIMA["MSE"].iloc[0]), '.2f')
            },
            ignore_index=True)

        df = df.append(
            {
                "Model": "SARIMA",
                "Size": "All",
                "Distance": "-",
                "Generation": "-",
                "RMSE": format((float)(dfSARIMA["RMSE"].iloc[0]), '.2f'),
                "MAPE": format((float)(dfSARIMA["MAPE"].iloc[0]), '.2f'),
                "MAE": format((float)(dfSARIMA["MAE"].iloc[0]), '.2f'),
                "MSE": format((float)(dfSARIMA["MSE"].iloc[0]), '.2f')
            },
            ignore_index=True)
        for model in [ModelType.DCRNN, ModelType.LSTM, ModelType.STCONV]:
            for datasetsize in [DatasetSize.Experimental, DatasetSize.Tiny, DatasetSize.Small, DatasetSize.Medium]:
                for GenerationType in ["Generative", "Manual", "LR"]:
                    for distanceType in DistanceType:
                        if ((datasetsize == DatasetSize.Small or datasetsize == DatasetSize.Medium) and (GenerationType == "Manual" or GenerationType == "LR")):
                            continue
                        if not (GenerationType == "Generative"):
                            datasetName = f"{datasetsize.name}{GenerationType}"
                        else:
                            datasetName = datasetsize.name
                        if (model == ModelType.LSTM):
                            modelName = "GCLSTM"
                        else:
                            modelName = model.name
                        dfResultsTemp = dfResults[dfResults["Model"]
                                                  == model.name]
                        dfResultsTemp = dfResultsTemp[dfResultsTemp["Size"]
                                                      == datasetName]
                        dfResultsTemp = dfResultsTemp[dfResultsTemp["DistanceType"]
                                                      == distanceType.name]
                        dfResultsTemp = dfResultsTemp[["Criterion", "Loss"]]
                        dfResultsTemp = dfResultsTemp.groupby(
                            ["Criterion"]).min().T
                        df = df.append(
                            {
                                "Model": modelName,
                                "Size": datasetsize.name,
                                "Distance": distanceType.name,
                                "Generation": GenerationType,
                                "RMSE": format(
                                    (float)(dfResultsTemp["RMSE"].iloc[0]), '.2f'),
                                "MAPE": format(
                                    (float)(dfResultsTemp["MAPE"].iloc[0]), '.2f'),
                                "MAE": format((float)(dfResultsTemp["MAE"].iloc[0]), '.2f'),
                                "MSE": format((float)(dfResultsTemp["MSE"].iloc[0]), '.2f')
                            },
                            ignore_index=True)

        layout = dict(height=DataViz.calc_table_height(df) + 300)
        fig = go.Figure(data=[
            go.Table(header=dict(values=list(df.columns),
                                 fill_color='paleturquoise',
                                 align='left'),
                     cells=dict(values=[
                         df.Model, df.Size, df.Distance, df.Generation, df.RMSE, df.MAPE, df.MAE, df.MSE
                     ],
                fill_color='lavender',
                align='left'))
        ], layout=layout)

        fig.write_image(path_save)

    def calc_table_height(df, base=208, height_per_row=20, char_limit=30, height_padding=16.5):
        '''
        df: The dataframe with only the columns you want to plot
        base: The base height of the table (header without any rows)
        height_per_row: The height that one row requires
        char_limit: If the length of a value crosses this limit, the row's height needs to be expanded to fit the value
        height_padding: Extra height in a row when a length of value exceeds char_limit
        '''
        total_height = 0 + base
        for x in range(df.shape[0]):
            total_height += height_per_row
        for y in range(df.shape[1]):
            if len(str(df.iloc[x][y])) > char_limit:
                total_height += height_padding
        return total_height

    def BoxPlotResultsNonGNN(self, model: ModelType) -> None:
        if model == ModelType.LinearRegression:
            df = self.dfLR
        elif model == ModelType.ARIMA:
            df = self.dfARIMA
        elif model == ModelType.SARIMA:
            df = self.dfSARIMA
        for criterion in LossFunction.Criterions():
            dfTemp = df[df["Criterion"] == criterion.__name__]
            fig = px.box(dfTemp, x="Criterion", y="Loss")
            if not os.path.isfile(
                    os.path.join(
                        self.path_save_plots,
                        f"boxplot_result_{model.name}_{criterion.__name__}.png")):
                fig.write_image(
                    os.path.join(
                        self.path_save_plots,
                        f"boxplot_result_{model.name}_{criterion.__name__}.png"))

    def BoxPlotResults(self, distanceType : DistanceType) -> None:
        dfSTCONV = self.dfSTCONV
        dfLSTM = self.dfLSTM
        dfDCRNN = self.dfDCRNN
        dfSTCONV["Type"] = "STCONV"
        dfLSTM["Type"] = "GCLSTM"
        dfDCRNN["Type"] = "DCRNN"
        df = dfSTCONV.append(dfLSTM, ignore_index=True)
        df = df.append(dfDCRNN, ignore_index=True)
        df = df[df["DistanceType"] == distanceType.name]
        df = df[df["Loss"] < 20]
        for criterion in LossFunction.Criterions():
            dfTemp = df[df["Criterion"] == criterion.__name__]
            fig = px.box(dfTemp, x="Size", y="Loss", color="Type")
            filename = os.path.join(self.path_save_plots,f"boxplot_result_{criterion.__name__}_{distanceType.name}.png")
            if not os.path.isfile(filename):
                fig.write_image(filename)

    def RegressionLRTruePredicted(self, datareader: DataReader) -> None:

        parameters = {
            'normalize': [True],
        }

        path_save_LR = os.path.join(self.path_save_plots,
                                    "RegressionLRTruePredicted")

        if not os.path.exists(path_save_LR):
            os.makedirs(path_save_LR)

        lr_model = LinearRegression()
        clf = GridSearchCV(lr_model, parameters, refit=True, cv=5)
        for index, (X_train, X_test, Y_train, Y_test,
                    node_id) in enumerate(LinearRegressionDataset(datareader)):
            best_model = clf.fit(X_train, Y_train)
            y_pred = best_model.predict(X_test)
            Y_test = [item for sublist in Y_test.tolist() for item in sublist]
            y_pred = [item for sublist in y_pred.tolist() for item in sublist]
            dict1 = dict(Time=np.arange(len(y_pred)),
                         Actual=Y_test,
                         Predicted=y_pred)
            df = pd.DataFrame(dict1)
            fig = px.line(df,
                          x='Time',
                          y=["Actual", "Predicted"],
                          title="Prediction for node {0}".format(node_id))
            fig.write_image(
                os.path.join(path_save_LR, "{0}.png".format(node_id)))
            print("Regression LR True Predicted node {0}".format(node_id))
            break

    def GNNRegresionTrainLoss(self) -> None:

        for size in DatasetSize:
            if size == DatasetSize.All: continue
            for model in [ModelType.DCRNN, ModelType.STCONV, ModelType.LSTM]:
                path_save = os.path.join(
                    self.path_save_plots, f"MAE_{model.name}_{size.name}.png")
                if os.path.isfile(path_save):
                    continue
                dfPlot = pd.DataFrame()
                if model == ModelType.DCRNN:
                    df = self.dfDCRNN
                elif model == ModelType.STCONV:
                    df = self.dfSTCONV
                elif model == ModelType.LSTM:
                    df = self.dfLSTM

                dfTemp = df[df["Criterion"] == "MAE"]
                dfTemp = dfTemp[dfTemp["Size"] == size.name]
                dfTemp2 = dfTemp[["Criterion", "Loss"]]
                dfTemp2 = dfTemp2.groupby(["Criterion"]).min()
                minLoss = dfTemp2["Loss"].iloc[0]
                BestTrial = dfTemp[dfTemp["Loss"] == minLoss]
                df = dfTemp[dfTemp["Trial"] ==
                            BestTrial["Trial"].iloc[0]]
                datalist = df["Loss"].tolist()
                dfPlot["Loss"] = datalist
                dfPlot["Time"] = np.arange(1,len(datalist) + 1)

                fig = px.line(dfPlot, x="Time", y="Loss")

                fig.write_image(path_save)

    def HeatMapLoss(self, model : ModelType) -> None:
        name_save = f"HeatMap{model.name}.png"
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return
        if model == ModelType.LinearRegression:
            dfInfo = self.dfLR
        elif model == ModelType.ARIMA:
            dfInfo = self.dfARIMA
        elif model == ModelType.SARIMA:
            dfInfo = self.dfSARIMA
        dfMeta = self.dfMeta
        dfInfo = dfInfo[dfInfo["Criterion"] == "MAE"]
        # dfInfo = dfInfo[dfInfo["Loss"] < 20]
        df = pd.merge(dfMeta, dfInfo, left_on='ID', right_on='Node_Id')
        fig = px.scatter_mapbox(
            df,
            lat="Latitude",
            lon="Longitude",
            color="Loss",
            color_continuous_scale=px.colors.sequential.Bluered,
            zoom=8,
            mapbox_style="open-street-map",
            title=f'Map for Loss {model.name}')

        fig.write_image(path_save)

    def SigmaEpsilonTable(self, name_save: str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        dfSTCONV = self.dfSTCONV
        dfDCRNN = self.dfDCRNN
        dfLSTM = self.dfLSTM
        dfResults = dfSTCONV.append(dfDCRNN, ignore_index=True)
        dfResults = dfResults.append(dfLSTM, ignore_index=True)

        df = pd.DataFrame(columns=Constants.sigma_array,
                          index=Constants.epsilon_array)
        for sigma in Constants.sigma_array:
            for epsilon in Constants.epsilon_array:
                dfResultsTemp = dfResults[dfResults["Sigma"] == sigma]
                dfResultsTemp = dfResultsTemp[dfResultsTemp["Epsilon"] == epsilon]
                dfResultsTemp = dfResultsTemp.groupby(["Criterion"]).min()
                df.loc[epsilon][sigma] = format(
                    (float)(dfResultsTemp["Loss"].iloc[0]), '.2f')
        headers = list(df.columns)
        headers.insert(0, "")
        print(headers)
        fig = go.Figure(data=[
            go.Table(header=dict(values=headers,
                                 fill_color='paleturquoise',
                                 align='left'),
                     cells=dict(values=[
                         Constants.epsilon_array, df[1], df[3], df[5], df[10]
                     ],
                fill_color='lavender',
                align='left'))
        ])
        fig.write_image(path_save)
        # TODO

    def GeneralViz_Run(self, datareader: DataReader):

        # General Datavizualization
        self.BoxPlotSpeed("boxplot.png")
        self.MapPlotSensors("mapplotAll.png", DatasetSize.All)
        self.MapPlotSensors("mapplotExperimental.png",
                            DatasetSize.Experimental)
        self.MapPlotSensors("mapplotTiny.png", DatasetSize.Tiny)
        self.MapPlotSensors("mapplotSmall.png", DatasetSize.Small)
        self.MapPlotSensors("mapplotMedium.png", DatasetSize.Medium)
        self.PieChartRoadwayType("piechart.png")
        self.MapHeatmapSpeed("mapheat9.png", 9)
        self.MapHeatmapSpeed("mapheat15.png", 15)
        self.MapHeatmapSpeed("mapheat18.png", 18)
        self.MapHeatmapSpeed("mapheat22.png", 22)
        self.GraphHeatmap("graph.png", datareader)

    def Experiment_Run(self, datareader: DataReader):

        # self.TableFinalResults("tableresults.png")
        # self.BoxPlotResults(DistanceType.Geodesic)
        self.BoxPlotResults(DistanceType.OSRM)
        # self.BoxPlotResultsNonGNN(ModelType.LinearRegression)
        # self.BoxPlotResultsNonGNN(ModelType.ARIMA)
        # self.BoxPlotResultsNonGNN(ModelType.SARIMA)
        # self.RegressionLRTruePredicted(datareader)
        # self.HeatMapLoss(ModelType.LinearRegression)
        # self.HeatMapLoss(ModelType.ARIMA)
        # self.HeatMapLoss(ModelType.SARIMA)
        # self.SigmaEpsilonTable("SigmaEpsilonTable.png")
        # self.GNNRegresionTrainLoss()

    def ReadInfo():
        datareader = DataReader()
        dfInfo, dfMeta = datareader.visualization()
        return datareader, dfInfo, dfMeta

    def GeneralViz(datareader: DataReader, dfInfo: pd.DataFrame,
                   dfMeta: pd.DataFrame):
        path_save_plots = os.path.join(Folders.path_save_plots, "GeneralViz")

        if not os.path.exists(path_save_plots):
            os.makedirs(path_save_plots)

        dataviz = DataViz(path_data=Folders.path_data,
                          path_save_plots=path_save_plots,
                          path_processed_data=Folders.proccessed_data_path,
                          path_results=Folders.results_path,
                          graph_info_txt=Folders.graph_info_path,
                          dfInfo=dfInfo,
                          dfMeta=dfMeta,
                          dfLR=pd.DataFrame(),
                          dfARIMA=pd.DataFrame(),
                          dfSARIMA=pd.DataFrame(),
                          dfSTCONV=pd.DataFrame(),
                          dfLSTM=pd.DataFrame(),
                          dfDCRNN=pd.DataFrame())

        dataviz.GeneralViz_Run(datareader)

    def Experiment(datareader: DataReader, dfInfo: pd.DataFrame,
                   dfMeta: pd.DataFrame):
        for experiment in os.listdir(Folders.results_ray_path):
            path_save_plots = os.path.join(Folders.path_save_plots, experiment)

            if not os.path.exists(path_save_plots):
                os.makedirs(path_save_plots)

            dfLR, dfARIMA, dfSARIMA, dfSTCONV, dfLSTM, dfDCRNN = datareader.results(
                experiment)
            dataviz = DataViz(path_data=Folders.path_data,
                              path_save_plots=path_save_plots,
                              path_processed_data=Folders.proccessed_data_path,
                              path_results=Folders.results_path,
                              graph_info_txt=Folders.graph_info_path,
                              dfInfo=dfInfo,
                              dfMeta=dfMeta,
                              dfLR=dfLR,
                              dfARIMA=dfARIMA,
                              dfSARIMA=dfSARIMA,
                              dfSTCONV=dfSTCONV,
                              dfLSTM=dfLSTM,
                              dfDCRNN=dfDCRNN)

            dataviz.Experiment_Run(datareader)

    def Run():
        datareader, dfInfo, dfMeta = DataViz.ReadInfo()
        # DataViz.GeneralViz(datareader,dfInfo,dfMeta)
        DataViz.Experiment(datareader, dfInfo, dfMeta)
