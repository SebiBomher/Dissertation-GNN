from Scripts.data_proccess import get_dataset,create_proccessed_data, get_graph_info, get_proccess_data


path_data = "Data"
path_proccess_data = "Proccessed"
graph_info_txt = "d07_text_meta_2021_03_27.txt"
create_proccessed_data(path_data,path_proccess_data,graph_info_txt)
# train_dataset, validation_dataset, test_dataset, num_nodes = get_dataset(path=path_data, train_test_ratio = 0.8, train_validation_ratio = 0.8,batch_size=16,time_steps=1,epsilon=0.5,lamda=10)