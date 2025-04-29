#from Training_baseline import train
from Training import train

"""
main script

"""

# data_dir: the directory in which datasets are stored
data_dir = '../../Files/'
epochs = 200
units = 6 # number of model's units
mini_batch_size = 600*4 # mini_batch size
batch_size = 1 # batch size
lr = 3e-4 # initial learning rate

name_model = ''

comp = 'CL1B'
dataset = ''

models = ['LSTM', 'ED', 'S4D', 'S6'] 
#models = ['LSTMb', 'EDb'] 
for model in models:
      train(data_dir=data_dir,
            save_folder=model+comp+dataset+name_model,
            dataset=comp+'/'+comp+dataset,
            comp=comp,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            learning_rate=lr,
            units=units,
            epochs=epochs,
            model=model,
            inference=False)
