import yfinance as yf 
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import time
import gc


import torch.optim as optim
from Functions.utils import  accuracy

class MyDataset(torch.utils.data.Dataset):
  def __init__(self , studied_val): # stage = phase de l'entraînement
    super().__init__()
    def Vix_treatment(data_indicator):
        data_indicator = data_indicator.drop(columns = ['Adj Close','Volume'])

        new_col_name = {}
        for col in data_indicator.columns :
                    new_col_name[col] =   'VIX'+ '_' + col
        data_indicator.rename(columns=new_col_name,inplace=True)
        return data_indicator

    def avg_compute (df, mean_period, dropna = True):
        origin_df = df
        for periode in mean_period :

            reduced_mean = origin_df.rolling(periode).mean()
            new_col_name = {}

            for col in reduced_mean.columns :
                new_col_name[col] = 'avg_'+ col + '_' + str(periode)

            reduced_mean.rename(columns=new_col_name,inplace=True)

            df= pd.concat([df, reduced_mean], axis =1)
        if dropna :
            return(df.dropna())

        return(df)

    def value_split (Values, Mixed_df, Other_data, Periodes):
        Value_df_list = {}
        scaler = MinMaxScaler()
        for val in Values :

            temp_df = Mixed_df.xs(val, axis=1, level=1, drop_level=True)
            temp_df.drop(columns = 'Adj Close')
            temp_df = pd.concat([temp_df, Other_data], axis =1)
            temp_df = avg_compute(temp_df,Periodes)
            scaled_features_df = scaler.fit_transform(temp_df)
            temp_df = pd.DataFrame(scaled_features_df,columns=temp_df.columns)
    #         temp_df = pd.DataFrame(scaled_features_df, index = temp_df.index, columns=temp_df.columns)

            Value_df_list[val] = temp_df
        return Value_df_list



    def data_fetch (Periode, Interval, Periodes, Values = studied_val ):
        tickers_names = ""
        for val in Values :
            tickers_names += " "+val 

        data = yf.download(tickers = tickers_names,  # list of tickers
                    period =Periode,         # time period
                    interval = Interval,       # trading interval
                    prepost = False,       # download pre/post market hours data?
                    repair = True)  
        data_indicator = yf.download(tickers = "^VIX",  # list of tickers
                    period = Periode,         # time period
                    interval = Interval,       # trading interval
                    prepost = False,       # download pre/post market hours data?
                    repair = True) 

        data_indicator = Vix_treatment(data_indicator)
        extended_data = value_split(Values,data,data_indicator,Periodes)


        return(extended_data)
    
    raw_data_df = data_fetch (Periode="5y", Interval="1d", Periodes = [30,60,100]) 
    
    len_seq = 5


    def split_sequence(df, n_steps):
        Input , Label = list(), list()
        for i in range(len(df)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(df)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = df.iloc[i:end_ix,:], df.iloc[end_ix:end_ix+1,:][["High", "Low"]]
    #         seq_x, seq_y = df[i:end_ix,:], df[end_ix:end_ix+1,:][["High", "Low"]]

            Input.append(seq_x)
            Label.append(seq_y)
        return Input, Label

    def data_parsing (df_dict, n_steps):
        Input_formated_data, Label_formated_data = list(), list()
        for value_df in df_dict.values():
            Input, Label = split_sequence(value_df, n_steps) 
            Input_formated_data=Input_formated_data +Input
            Label_formated_data=Label_formated_data+ Label
        return(Input_formated_data, Label_formated_data)

    self.Inputs, self.Labels =  data_parsing(raw_data_df, len_seq)


  def __len__(self):
    return (len(self.Inputs))
  
  def __getitem__(self,idx):
    return self.Inputs[idx].to_numpy(),self.Labels[idx].to_numpy()




st = time.time()
batch_size = 32
dataset_training = MyDataset(studied_val= ["SPY", "AAPL", "INTC", "JPM", "WMT"])
# dataset_training = MyDataset(studied_val= ["SPY", "AAPL", "CAGR", "DAST", "LVMH", "AMZN", "INTC"])

dataset_validation = MyDataset(studied_val= ["MSFT", "AMZN"])

train_data = torch.utils.data.DataLoader(dataset_training, batch_size=batch_size, shuffle=True)
val_data = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)

et = time.time()

print("Data processing took ", et-st, 'second')


class LSTMnet (nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(40, 256, 5, batch_first =True)
        self.lstm2 = nn.LSTM(256, 8, batch_first =True)
        self.FC = nn.Linear(8, 4)
        self.FC2 = nn.Linear(4, 2)
        self.softmax = nn.LogSoftmax(1)


        self.tanh  = nn.Tanh()
        # self.Flatten = torch.nn.Flatten()


    def forward(self, input):
        x,(y1,y2) = self.lstm1(input)
        x, (y1,y2) = self.lstm2(x)
        out = x[:, -1]
        out= self.FC(out)
        out= self.FC2(out)

        return self.softmax(out)


model = LSTMnet()
model = model.double()
optimizer = optim.Adam(model.parameters(), lr=0.5)
loss_f = nn.MSELoss()


def training(model, epoch_number, GPU ): # seulement pour les CPU  => TODO Passer en mode GPU.

  epoch_training_acc, epoch_training_loss, epoch_val_acc, epoch_val_loss = [],[],[],[]
  if GPU :
    print('trying to train on GPU' )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Device found : ', device)
  
  if torch.cuda.is_available() and GPU:
    nbr_GPU = torch.cuda.device_count()
    crt_gpu = torch.cuda.current_device()
    crt_gpu_name = torch.cuda.torch.cuda.get_device_name(crt_gpu)
    model.to(device)
    print('currently training on '+str(nbr_GPU)+' GPU '+ 'GPU type : '+str(crt_gpu_name) )

  for epoch in range(epoch_number):

    st = time.time()

    model.train()

    train_loss, running_train_acc = 0,0

    for local_batch, local_labels in train_data:

      if GPU:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
      
      y_pred = model(local_batch)

      loss = loss_f(y_pred, torch.squeeze(local_labels))

      loss.backward()
      train_loss += loss.item()

      running_train_acc += accuracy(y_pred, local_labels)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      local_batch, local_labels, y_pred = None, None, None
      del local_batch, local_labels, y_pred

    train_loss /= len(train_data)
    running_train_acc /= len(train_data)

    epoch_training_acc.append(running_train_acc)
    epoch_training_loss.append(train_loss)

    # writer.add_scalar("train/Train_acc", running_train_acc, epoch)
    # writer.add_scalar("train/Train_loss", train_loss, epoch)

    batchs_acc, train_loss = None, None
    del batchs_acc, train_loss
    gc.collect()

    #evaluation :
    model.eval()
    val_loss, running_val_acc = 0,0

    for local_batch, local_labels in val_data:
      
      if GPU:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

      y_pred = model(local_batch)
      loss = loss_f(y_pred, torch.squeeze(local_labels))
      val_loss += loss.item()
      running_val_acc += accuracy(y_pred, torch.squeeze(local_labels))

    local_batch, local_labels,y_pred = None, None , None
    del local_batch, local_labels, y_pred

    val_loss /= len(val_data)
    running_val_acc /= len(val_data)

    epoch_val_acc.append(running_val_acc)
    epoch_val_loss.append(val_loss)

    # writer.add_scalar("Validation/Validation_acc", running_val_acc, epoch)
    # writer.add_scalar("Validation/Validation_loss", val_loss, epoch)

    running_val_acc, val_loss = None, None
    del running_val_acc, val_loss

    et = time.time()
    print(epoch+1, '/', epoch_number , " took ", et-st, 'second')
    print('------ training accuracy :',"{:.3f}".format(epoch_training_acc[epoch]), ' | validation accuracy :',"{:.3f}".format(epoch_val_acc[epoch]), ' | training loss :',"{:.3f}".format(epoch_training_loss[epoch]), ' | valildation loss :', "{:.3f}".format(epoch_val_loss[epoch]))
    
    batchs_acc, validation_loss,loss = None, None,None
    del batchs_acc, validation_loss,loss
    gc.collect()

  return (epoch_training_loss, epoch_training_acc , epoch_val_loss , epoch_val_acc)

epoch_training_loss, epoch_training_acc , epoch_val_loss , epoch_val_acc = training(model,2000, False)
