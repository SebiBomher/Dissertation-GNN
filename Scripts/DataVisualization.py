#region Imports

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
from Scripts.Utility import Folders
from Scripts.DataProccess import DataReader, DatasetSize, DatasetSizeNumber, Graph

#endregion


class DataViz():
    r"""
        Class used to create different plots for Data Visualization. It covers dataset data visualization and results data visualization
    """

    def __init__(self,
                 path_data: str,
                 graph_info_txt: str,
                 path_save_plots: str,
                 path_processed_data: str,
                 path_results: str,
                 dfInfo: pd.DataFrame,
                 dfMeta: pd.DataFrame,
                 dfLR: pd.DataFrame,
                 dfARIMA: pd.DataFrame,
                 dfSARIMA: pd.DataFrame,
                 dfSTCONV: pd.DataFrame,
                 dfDCRNN: pd.DataFrame,
                 dfCUSTOM: pd.DataFrame):

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
        self.dfCUSTOM = dfCUSTOM
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
        df['Day'] = df['Timestamp'].apply(
            lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M:%S').weekday())
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
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="ID", hover_data=["Type", "Lanes"],
                                color_discrete_sequence=["black"], zoom=zoom, size_max=15, mapbox_style="open-street-map")

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
        dataframe = pd.DataFrame(
            {'Type': series.index, 'count': series.values})
        dataframe = dataframe.replace({'CD': 'Coll/Dist', 'CH': 'Conventional Highway',
                                       'FF': 'Freeway-Freeway connector', 'FR': 'Off Ramp', 'HV': 'HOV', 'ML': 'Mainline', 'OR': 'On Ramp'})
        fig = px.pie(dataframe, values='count',
                     names='Type', title='Types of roads')

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
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Speed",
                                color_continuous_scale=px.colors.sequential.Bluered, zoom=8, mapbox_style="open-street-map", title='Traffic speed at {0}:00'.format(str(hour)))

        fig.write_image(path_save)

    def GetGraphMatrixFromGraph(graph : Graph, size : DatasetSize):
        nr_nodes = Graph.get_number_nodes_by_size(size)
        graph_matrix = np.zeros((nr_nodes, nr_nodes))
        for index, (edge_index, edge_weight) in enumerate(zip(np.transpose(graph.edge_index), graph.edge_weight)):
            graph_matrix[edge_index[0]
                        ][edge_index[1]] = edge_weight
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
                for dataset in DatasetSize:
                    if dataset != DatasetSize.ExperimentalManual and dataset != DatasetSize.ExperimentalLR and dataset != DatasetSize.TinyManual and dataset != DatasetSize.TinyLR and dataset != DatasetSize.All:  
                        graph = Graph(epsilon,sigma, dataset, datareader)
                        graph_matrix = DataViz.GetGraphMatrixFromGraph(graph,dataset)
                        fig = px.imshow(graph_matrix, title='Graph Heatmap for epsilon {0} and sigma {1} with size {2}'.format(
                            epsilon, sigma, dataset.name))

                        fig.write_image("{0}_{1}_{2}_{3}.png".format(
                            path_save, epsilon, sigma, dataset.name))
        for dataset in [DatasetSize.ExperimentalManual,DatasetSize.ExperimentalLR,DatasetSize.TinyManual,DatasetSize.TinyLR]:
            graph = Graph(0.1,1, dataset, datareader)
            graph_matrix = DataViz.GetGraphMatrixFromGraph(graph,dataset)
            fig = px.imshow(graph_matrix, title='Graph Heatmap for size {0}'.format(dataset.name))
            fig.write_image("{0}.png".format(dataset.name))

    def TableFinalResults(self, name_save: str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        dfLR = self.dfLR
        dfSTCONV = self.dfSTCONV
        dfCUSTOM = self.dfCUSTOM

        df = pd.DataFrame(
            columns=["Type", "Size", "RMSE", "MAPE", "MAE", "MSE"])

        dfLR = dfLR[["Criterion", "Loss"]]
        dfLR = dfLR.groupby(["Criterion"]).mean().T

        dfSTCONV = dfSTCONV[["Criterion", "Loss", "Size"]]
        dfCUSTOM = dfCUSTOM[["Criterion", "Loss", "Size"]]

        df = df.append({"Type": "Linear Regression", "Size": "All", "RMSE": format((float)(dfLR["RMSE"].iloc[0]), '.2f'), "MAPE": format((float)(
            dfLR["MAPE"].iloc[0]), '.2f'), "MAE": format((float)(dfLR["MAE"].iloc[0]), '.2f'), "MSE": format((float)(dfLR["MSE"].iloc[0]), '.2f')}, ignore_index=True)

        for datasetsize in DatasetSize:
            dfSTCONVTemp = dfSTCONV[dfSTCONV["Size"] == datasetsize.name]
            dfSTCONVTemp = dfSTCONVTemp[["Criterion", "Loss"]]
            dfSTCONVTemp = dfSTCONVTemp.groupby(["Criterion"]).min().T
            df = df.append({"Type": "STCONV",
                            "Size": datasetsize.name,
                            "RMSE": format((float)(dfSTCONVTemp["RMSE"].iloc[0]), '.2f'),
                            "MAPE": format((float)(dfSTCONVTemp["MAPE"].iloc[0]), '.2f'),
                            "MAE": format((float)(dfSTCONVTemp["MAE"].iloc[0]), '.2f'),
                            "MSE": format((float)(dfSTCONVTemp["MSE"].iloc[0]), '.2f')}, ignore_index=True)

        for datasetsize in DatasetSize:
            dfCUSTOMTemp = dfCUSTOM[dfCUSTOM["Size"] == datasetsize.name]
            dfCUSTOMTemp = dfCUSTOMTemp[["Criterion", "Loss"]]
            dfCUSTOMTemp = dfCUSTOMTemp.groupby(["Criterion"]).min().T
            df = df.append({"Type": "Custom",
                            "Size": datasetsize.name,
                            "RMSE": format((float)(dfCUSTOMTemp["RMSE"].iloc[0]), '.2f'),
                            "MAPE": format((float)(dfCUSTOMTemp["MAPE"].iloc[0]), '.2f'),
                            "MAE": format((float)(dfCUSTOMTemp["MAE"].iloc[0]), '.2f'),
                            "MSE": format((float)(dfCUSTOMTemp["MSE"].iloc[0]), '.2f')}, ignore_index=True)
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df.Type, df.Size, df.RMSE, df.MAPE, df.MAE, df.MSE],
                       fill_color='lavender',
                       align='left'))
        ])

        fig.write_image(path_save)

    def BoxPlotResultsLR(self) -> None:
        dfLR = self.dfLR
        dfLR = dfLR[dfLR["Loss"] < 100]
        for criterion in LossFunction.Criterions():
            dfTemp = dfLR[dfLR["Criterion"] == criterion.__name__]
            fig = px.box(dfTemp, x="Criterion", y="Loss")
            if not os.path.isfile(os.path.join(
                self.path_save_plots, "boxplot_result_LR_{0}.png".format(criterion.__name__))):
                fig.write_image(os.path.join(
                    self.path_save_plots, "boxplot_result_LR_{0}.png".format(criterion.__name__)))

    def BoxPlotResults(self) -> None:
        dfSTCONV = self.dfSTCONV
        dfCUSTOM = self.dfCUSTOM
        dfSTCONV["Type"] = "STCONV"
        dfCUSTOM["Type"] = "Custom"
        df = dfSTCONV.append(dfCUSTOM, ignore_index=True)
        for criterion in LossFunction.Criterions():
            dfTemp = df[df["Criterion"] == criterion.__name__]
            fig = px.box(dfTemp, x="Size", y="Loss", color="Type")
            if not os.path.isfile(os.path.join(
                self.path_save_plots, "boxplot_result_{0}.png".format(criterion.__name__))):
                fig.write_image(os.path.join(
                    self.path_save_plots, "boxplot_result_{0}.png".format(criterion.__name__)))

    def RegressionLRTruePredicted(self, datareader: DataReader) -> None:

        parameters = {
            'normalize': [True],
        }

        path_save_LR = os.path.join(self.path_save_plots, "RegressionLRTruePredicted")

        if not os.path.exists(path_save_LR):
            os.makedirs(path_save_LR)

        lr_model = LinearRegression()
        clf = GridSearchCV(lr_model, parameters, refit=True, cv=5)
        for index, (X_train, X_test, Y_train, Y_test, node_id) in enumerate(LinearRegressionDataset(datareader)):
            best_model = clf.fit(X_train, Y_train)
            y_pred = best_model.predict(X_test)
            Y_test = [item for sublist in Y_test.tolist() for item in sublist]
            y_pred = [item for sublist in y_pred.tolist() for item in sublist]
            dict1 = dict(Time=np.arange(len(y_pred)),
                         Actual=Y_test, Predicted=y_pred)
            df = pd.DataFrame(dict1)
            fig = px.line(df, x='Time', y=[
                          "Actual", "Predicted"], title="Prediction for node {0}".format(node_id))
            fig.write_image(os.path.join(path_save_LR, "{0}.png".format(node_id)))
            print("Regression LR True Predicted node {0}".format(node_id))
            break

    def RegressionLoss(self,  name_save: str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return
        dfCUSTOM = self.dfCUSTOM
        dfSTCONV = self.dfSTCONV
        dfPlot = pd.DataFrame()
        dfPlot["Time"] = np.arange(300)
        for size in DatasetSize:
            if size != DatasetSize.Medium:
                dfSTCONVTemp = dfSTCONV[dfSTCONV["Criterion"] == "MAE"]
                dfSTCONVTemp = dfSTCONVTemp[dfSTCONVTemp["Size"] == size.name]
                dfSTCONVTemp2 = dfSTCONVTemp[["Criterion", "Loss"]]
                dfSTCONVTemp2 = dfSTCONVTemp2.groupby(["Criterion"]).min()
                minLoss = dfSTCONVTemp2["Loss"].iloc[0]
                BEstTrial = dfSTCONVTemp[dfSTCONVTemp["Loss"] == minLoss]
                df = dfSTCONVTemp[dfSTCONVTemp["Trial"]
                                  == BEstTrial["Trial"].iloc[0]]
                datalist = df["Loss"].tolist()
                datalist.extend(np.zeros(300 - len(datalist)))
                dfPlot["STCONV_{0}".format(size.name)] = datalist

            dfCUSTOMTemp = dfCUSTOM[dfCUSTOM["Criterion"] == "MAE"]
            dfCUSTOMTemp = dfCUSTOMTemp[dfCUSTOMTemp["Size"] == size.name]
            dfCUSTOMTemp2 = dfCUSTOMTemp[["Criterion", "Loss"]]
            dfCUSTOMTemp2 = dfCUSTOMTemp2.groupby(["Criterion"]).min()
            minLoss = dfCUSTOMTemp2["Loss"].iloc[0]
            BEstTrial = dfCUSTOMTemp[dfCUSTOMTemp["Loss"] == minLoss]
            df = dfCUSTOMTemp[dfCUSTOMTemp["Trial"]
                              == BEstTrial["Trial"].iloc[0]]
            datalist = df["Loss"].tolist()
            datalist.extend(np.zeros(300 - len(datalist)))
            dfPlot["CUSTOM_{0}".format(size.name)] = datalist
        columns = dfPlot.columns.tolist()
        columns.pop(0)
        fig = px.line(dfPlot, x="Time", y=columns)
        fig.write_image(path_save)

    def HeatMapLoss(self, name_save: str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return
        dfInfo = self.dfLR
        dfMeta = self.dfMeta
        dfInfo = dfInfo[dfInfo["Criterion"] == "MAE"]
        dfInfo = dfInfo[dfInfo["Loss"] < 20]
        df = pd.merge(dfMeta, dfInfo, left_on='ID', right_on='Node_Id')
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Loss",
                                color_continuous_scale=px.colors.sequential.Bluered, zoom=8, mapbox_style="open-street-map", title='Map for Loss MAE')

        fig.write_image(path_save)

    def GNNNRegresion(self,name_save : str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        # TODO

        fig = px.box([], x="Size", y="Loss", color="Type")
        fig.write_image(path_save)
        return

    def HeatMapLossGNNEachModel(self,name_save : str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        # TODO

        fig = px.box([], x="Size", y="Loss", color="Type")
        fig.write_image(path_save)
        return

    def HeatMapLossGNNEachMapGeneration(self,name_save : str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        # TODO

        fig = px.box([], x="Size", y="Loss", color="Type")
        fig.write_image(path_save)
        return

    def BoxPlotLossOnEachNode(self,name_save : str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        # TODO

        fig = px.box([], x="Size", y="Loss", color="Type")
        fig.write_image(path_save)
        return

    def ResultsBestHyperParameters(self,name_save : str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        # TODO

        fig = px.box([], x="Size", y="Loss", color="Type")
        fig.write_image(path_save)
        return

    def HyperParametersUsedOnTrialDescending(self,name_save : str) -> None:
        path_save = os.path.join(self.path_save_plots, name_save)
        if os.path.isfile(path_save):
            return

        # TODO

        fig = px.box([], x="Size", y="Loss", color="Type")
        fig.write_image(path_save)
        return

    def Run():
        datareader = DataReader()
        dfInfo, dfMeta = datareader.visualization()
        path_save_plots = os.path.join(Folders.path_save_plots,"GeneralViz")

        if not os.path.exists(path_save_plots):
            os.makedirs(path_save_plots)

        dataviz = DataViz(path_data = Folders.path_data,
                            path_save_plots = path_save_plots,
                            path_processed_data = Folders.proccessed_data_path,
                            path_results = Folders.results_path,
                            graph_info_txt = Folders.graph_info_path,
                            dfInfo = dfInfo,
                            dfMeta = dfMeta,
                            dfLR = pd.DataFrame(),
                            dfSTCONV = pd.DataFrame(),
                            dfCUSTOM = pd.DataFrame())
        # General Datavizualization
        dataviz.BoxPlotSpeed("boxplot.png")
        dataviz.MapPlotSensors("mapplotAll.png",DatasetSize.All)
        dataviz.MapPlotSensors("mapplotExperimental.png",DatasetSize.Experimental)
        dataviz.MapPlotSensors("mapplotTiny.png",DatasetSize.Tiny)
        dataviz.MapPlotSensors("mapplotSmall.png",DatasetSize.Small)
        dataviz.MapPlotSensors("mapplotMedium.png",DatasetSize.Medium)
        dataviz.PieChartRoadwayType("piechart.png")
        dataviz.MapHeatmapSpeed("mapheat9.png",9)
        dataviz.MapHeatmapSpeed("mapheat15.png",15)
        dataviz.MapHeatmapSpeed("mapheat18.png",18)
        dataviz.MapHeatmapSpeed("mapheat22.png",22)
        dataviz.GraphHeatmap("graph.png", datareader)

        for experiment in os.listdir(Folders.results_ray_path):
            path_save_plots = os.path.join(Folders.path_save_plots,experiment)

            if not os.path.exists(path_save_plots):
                os.makedirs(path_save_plots)

            dfLR,dfARIMA,dfSARIMA,dfSTCONV,dfCUSTOM,dfDCRNN = datareader.results(experiment)
            dataviz = DataViz(path_data = Folders.path_data,
                                path_save_plots = path_save_plots,
                                path_processed_data = Folders.proccessed_data_path,
                                path_results = Folders.results_path,
                                graph_info_txt = Folders.graph_info_path,
                                dfInfo = dfInfo,
                                dfMeta = dfMeta,
                                dfLR = dfLR,
                                dfARIMA = dfARIMA,
                                dfSARIMA = dfSARIMA,
                                dfSTCONV = dfSTCONV,
                                dfDCRNN = dfDCRNN,
                                dfCUSTOM = dfCUSTOM)

            #Results Visualization
            dataviz.TableFinalResults("tableresults.png")
            dataviz.BoxPlotResults()
            dataviz.BoxPlotResultsLR()
            dataviz.RegressionLRTruePredicted(datareader)
            dataviz.HeatMapLoss("HeatMapLRLossMAE.png")
            dataviz.RegressionLoss("Training_no_medium.png")