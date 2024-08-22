from Training import train


data_dir = '../../Files/'
#data_dir = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/Compressors/Pickles/'########
epochs = [1, 200]
units = 6
batch_size = 600*4
lr = 3e-4

#model = 'LSTM'
#model = 'S4D'
#model = 'Mamba'
#model = 'ED-CNN'

#comp = 'LA2A'
#comp = 'CL1B'

#dataset = '_analog'
#dataset = '_digital_cond_din'
#dataset = '_digital_cond_din01'
#dataset = '_digital_cond_din02'
#dataset = '_digital_cond_din005'
#dataset = '_digital_cond_time01'
#dataset = '_digital_cond_time02'
#dataset = '_digital_cond_time005'
#dataset = '_digital'

#dataset = '_digital_cond001'
#dataset = '_digital_cond01'
#dataset = '_digital_cond005'


name_model = ''

comp = 'CL1B'
dataset = '_digital_cond_time005'

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
