#from Training_baseline import train
from Training import train


data_dir = '../../Files/'
epochs = [1, 200]
units = 6
batch_size = 600*4
lr = 3e-4

name_model = ''

comp = 'CL1B'
dataset = ''

model = 'S4D'
train(data_dir=data_dir,
      save_folder=model+comp+dataset+name_model,
      dataset=comp+'/'+comp+dataset,
      comp=comp,
      batch_size=batch_size,
      learning_rate=lr,
      units=units,
      epochs=epochs,
      model=model,
      inference=False)
