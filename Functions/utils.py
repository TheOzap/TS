#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime 
from datetime import datetime
import torch
import os
import random, os
import numpy as np

import torch.optim as optim


##########################################################
## cette fonction sauvegarde les sortie de notre modèle ##
##########################################################

def output_save (dataFrame, output_folder, file_name ='out'): # TODO replace the index by batch nbr 

  if dataFrame.index[0]!=1 :
     dataFrame.index += 1-dataFrame.index[0] # on ramène les indice à partir de 1.

  now = datetime.now()
  date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
  dataFrame.to_csv(output_folder + '/' + file_name + '_' + date_time + '.csv')




###########################################################
## cette fonction sauvegarde notre modèle au format .zip ##
###########################################################

def save_model(model, dim,dt=datetime.now().strftime("%m-%d-%Y_%H:%M:%S"),output_folder='./TrainedModels'):
  model.eval()
  outfile = os.path.join(output_folder, 'model'+dt+'.zip')
  print(outfile)
  input = torch.zeros( dim[0], dim[1], dim[2],dim[3])   
  print(input.size())  
  m = torch.jit.trace(model, input)
  # Save to file
  torch.jit.save(m, outfile)

############################################################################
## Cette fonction plot la loss et l'accuracy de notre model au format png ##
##                                                                        ##
## En entrée il doit être fourni un dataframe ainsi qu'un path vers la ou ##
## On souhaite enregistrer notre png                                      ##
############################################################################

def plot_out_to_png (dataFrame, save_folder = './Results'):
  fig = plt.figure(constrained_layout=True, figsize = (16, 12))
  fig.add_subplot(211)
  plt.title("Losses :")
  plt.grid(color='k', linestyle='-', linewidth=1)
  plt.plot(dataFrame['train_loss'])
  plt.plot(dataFrame['val_loss'])

  fig.add_subplot(212)
  plt.title("Accuracies :")
  plt.grid(color='k', linestyle='-', linewidth=1)
  plt.plot(dataFrame['train_acc'])
  plt.plot(dataFrame['val_acc'])
  fig.savefig(save_folder + '/' +'perf.png')


###################################################################
## This function ensure the reproductibility of our experiences. ##
###################################################################

def seed_everything(seed: int):
   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#######################################
## cette fonction calcul d'accuracy  ##
#######################################

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    labels = torch.argmax(torch.squeeze(labels), dim=1)
    return torch.mean((classes == labels).float()).item()*100


# #######################################################################
# ## Cette fonction permet de selectionner un model parmis nos modèles ##
# #######################################################################

# def model_selector (model_name):
#     if model_name == "AlexNet" :
#         model = AlexNet()
#         return(model)
#     if model_name == "ModelTest" :
#         model = ConvNet()
#         return(model)
#     else :
#        return (0)



# class FullModel(nn.Module):

#     def __init__(self, model, softmax):
#         if softmax:
#           last_layer = nn.Softmax(dim=1)
#         else:
#           last_layer = lambda x: x
#         super(FullModel, self).__init__()
#         self.model = model
#         self.last_layer = last_layer

#     def forward(self, x):

#         return self.last_layer(self.model(x))

#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤#
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤ HYPERPARAMETER SELECTION ¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤#
#¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤¤#

#Faire un fichier de config 

def Optimizer_selection (params ,name, lr, weight_decay, GPU, cyclical = False, max_lr=0,min_lr=0, batch_p_cycle = 0):
    
    if name =='Adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name=='RMSprop':
        optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif name=='Adagrad':
        optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif name=='AdamW':
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, nesterov=True, momentum=0.9)
    
    if cyclical:
      val_p_c =batch_p_cycle//2
      scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, cycle_momentum=True, step_size_up=val_p_c)
      return(scheduler)

    if GPU :
      for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    return(optimizer)

def loss_selector(loss_name):
  # Il s'agit de la perte la plus couramment utilisée pour la classification multiclasse.
  # Elle mesure la différence entre les prédictions et les vraies étiquettes.
  if loss_name == 'CrossEntropy' :
    loss = nn.CrossEntropyLoss()
  # Cette perte est souvent utilisée pour les problèmes de régression où la valeur de sortie est continue.
  # Elle mesure la différence au carré entre les prédictions et les vraies étiquettes.
  elif loss_name == 'MSE':
    loss = nn.MSELoss()
  # Cette perte est utilisée pour la classification binaire.
  # Elle mesure la différence entre les prédictions et les vraies étiquettes binaires.
  elif loss_name == 'BinaryCrossEntropy':
    loss = nn.BCELoss()
  # Cette perte est utilisée pour la classification binaire avec des marges souples.
  # Elle mesure la différence entre les prédictions et les marges souhaitées.
  elif loss_name == 'HingeLoss':
    loss = nn.HingeEmbeddingLoss()
  # Cette perte est utilisée pour la classification multiclasse.
  # Elle mesure la distance entre deux distributions de probabilité, en comparant les prédictions et les vraies étiquettes.
  elif loss_name == 'Kullback-Leibler':
    loss =  nn.KLDivLoss()
  # Cette perte est une alternative à la perte de moyenne quadratique pour les problèmes de régression.
  # Elle est moins sensible aux valeurs aberrantes.
  elif loss_name == 'Huber loss':
    loss = nn.HuberLoss()
  # Cette perte est également une alternative à la perte de moyenne quadratique pour les problèmes de régression.
  # Elle est moins sensible aux valeurs aberrantes et peut être plus rapide à converger.
  elif loss_name == 'SmoothL1Loss':
    loss = nn.SmoothL1Loss()
  else :
      return(print('Unknown loss'))
  return (loss)