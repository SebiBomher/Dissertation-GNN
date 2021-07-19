from Scripts.models import ST_GCN
from Scripts.learn import train_val_and_test,MSE
from Scripts.data_proccess import get_dataset, get_dataset_experimental
from torch.nn import MSELoss,L1Loss
from torch.optim import Adam,SGD,RMSprop,Adamax

# Not Relevant for hypertuning, will implement regularization
learning_rate = 0.01
# [8,16,32,64]
batch_size = 16
# [1,3,6,9]
time_steps = 1
# [8,16,32,64,128]
hidden_channels = 32
#FIXED, Not Relevant for hypertuning
num_nodes = 8
#FIXED, Not Relevant for hypertuning
num_features = 3
# [1,3,5,7]
kernel_size = 1
# [1,3,5,7]
K = 1
# Not Relevant for hypertuning, will implement regularization
nb_epoch = 200
# MSE,MAE,MAPE,RMSE
# Not Relevant for hypertuning
criterion = MSELoss()
# Adam,SGD,RMSprop,Adamax
optimizer_type = "Adam"

train_dataset, validation_dataset, test_dataset = get_dataset_experimental(path="Data", train_test_ratio = 0.8,train_validation_ratio = 0.8,batch_size=batch_size,time_steps=time_steps)

model = ST_GCN(node_features = num_features,
                    num_nodes = num_nodes,
                    hidden_channels = hidden_channels,
                    kernel_size = kernel_size,
                    K = K)

optimizer = Adam(model.parameters(), lr=learning_rate)
 
train_val_and_test(model,train_dataset,validation_dataset,test_dataset,optimizer,nb_epoch,MSE)