from TrainingTCN import train

"""
main script

"""


# data_dir: the directory in which datasets are stored
data_dir = '../../Files/'
epochs = 200
units = 6 # number of model's units
batch_size = 1 # batch size
lr = 3e-4 # initial learning rate


name_model = ''

      
comp = 'CL1B'
dataset = ''
model = 'TCN'

train(data_dir=data_dir,
      save_folder=model+comp+dataset+name_model,
      dataset=comp+'/'+comp+dataset,
      comp=comp,
      half=False,
      batch_size=batch_size,
      learning_rate=lr,
      units=units,
      epochs=epochs,
      model=model,
      inference=True)