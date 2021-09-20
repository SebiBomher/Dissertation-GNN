import pandas as pd
from Scripts.data_proccess import DataReader
import plotly.express as px
import os


def BoxPlotSpeed(df : pd.DataFrame):
    # TODO
    fig = px.box(df, x="time", y="total_bill")
    fig.show()
    return

def MapPlotSensors():
    # TODO
    return

def PieChartRoadwayType(df : pd.DataFrame, path_save : str):
    # TODO
    series = df['LaneType'].value_counts(ascending=False,dropna=True)
    dataframe = pd.DataFrame({'LaneType':series.index, 'count':series.values})
    fig = px.pie(dataframe, values='count', names='LaneType', title='Types of roads')
    fig.show()
    fig.write_image(path_save)
    return

def BarPlostSamplesObserved():
    # TODO
    return

def MapHeatmapSpeed():
    # TODO
    return

def GraphHeatmap():
    # TODO
    return

def TableFinalResults():
    # TODO
    return

if __name__ == '__main__':
    # TODO
    num_samples = 16
    path_data = "E:\\FacultateMasterAI\\Dissertation-GNN\\Data"
    graph_info_txt = "d07_text_meta_2021_03_27.txt"
    path_save_plots = "E:\\FacultateMasterAI\\Dissertation-GNN\\Plots"
    if not os.path.exists(path_save_plots):
        os.mkdir(path_save_plots)
    datareader = DataReader(path_data,graph_info_txt)
    dataframe = datareader.visualization()
    PieChartRoadwayType(datareader,os.path.join(path_save_plots,"piechart.png"))