import pandas as pd
from Scripts.data_proccess import DataReader, DatasetSize, Graph
import plotly.express as px
import os
import datetime

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

def BarPlotSamplesObserved(df : pd.DataFrame, path_save : str) -> None:
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    fig.write_image(path_save)

def MapHeatmapSpeed(dfInfo : pd.DataFrame,dfMeta : pd.DataFrame, path_save : str, hour : int) -> None:
    # TODO
    zoom = 8
    dfInfo2 = dfInfo[dfInfo['Speed'].notna()]
    dfInfo2['Hour'] = dfInfo2['Timestamp'].apply(lambda x: datetime.datetime.strptime(x,'%m/%d/%Y %H:%M:%S').hour())
    dfInfo2 = dfInfo2[['Hour','Station','Speed']]
    dfInfo2 = dfInfo2.groupby(['Hour','Station']).mean()
    dfInfo2.loc[dfInfo2['Hour'] == hour]
    fig = px.scatter_mapbox(dfMeta, lat="Latitude", lon="Longitude", color="AvgSpeed",
    color_continuous_scale=px.colors.cyclical.IceFire, zoom=zoom, mapbox_style="open-street-map")
    fig.show()
    fig.write_image(path_save)

def GraphHeatmap(df : pd.DataFrame, path_save : str) -> None:
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
    path_processed_data = "D:\\FacultateMasterAI\\Dissertation-GNN\\Proccessed"
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
    # BarPlotSamplesObserved(dfInfo,os.path.join(path_save_plots,"barplot.png"))
    MapHeatmapSpeed(dfMeta,os.path.join(path_save_plots,"mapheat.png"))
    # GraphHeatmap(dfMeta,os.path.join(path_save_plots,"graph.png"))
    # TableFinalResults(dfMeta,os.path.join(path_save_plots,"tablefinal.png"))