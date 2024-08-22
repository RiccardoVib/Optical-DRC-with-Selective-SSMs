import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from Models import create_model_S4D, create_model_LSTM9, create_model_Mamba, create_model_ED_CNN
from DatasetsClass import DataGeneratorPicklesCL1B, DataGeneratorPicklesLA2A
import numpy as np
import random
from Metrics import ESR, RMSE
import sys
import time
from Losses import STFT, STFT_loss
from Utils import has_numbers

def train(**kwargs):
    batch_size = kwargs.get('batch_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    units = kwargs.get('units', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    comp = kwargs.get('comp', None)
    n_pickles = kwargs.get('n_pickles', 1)
    model_name = kwargs.get('model', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', [1, 60])

    epochs0 = epochs[0]
    epochs1 = epochs[1]
    

    #####seed
    np.random.seed(422)
    tf.random.set_seed(422)
    random.seed(422)

    if comp == 'CL1B':
        data_generator = DataGeneratorPicklesCL1B
    elif comp == 'LA2A':
        data_generator = DataGeneratorPicklesLA2A

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    #tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    fs = 48000
    w = 64
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, 0.)

    #
    dataset_test = dataset
    while has_numbers(dataset_test[8:]):
        dataset_test = dataset_test[:-1]

    # load the datasets
    train_gen = data_generator(data_dir, dataset + '_train.pickle', input_size=w, n_pickles=n_pickles, batch_size=batch_size)
    test_gen = data_generator(data_dir, dataset_test + '_test.pickle', input_size=w, n_pickles=n_pickles, batch_size=batch_size)

    training_steps = train_gen.training_steps*n_pickles
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps, epochs1), clipnorm=1)
    
    # create the model
    if model_name == 'Mamba':
        model = create_model_Mamba(b_size=batch_size, input_dims=w, model_input_dims=units//2, model_states=units, comp=comp)
    elif model_name == 'ED-CNN':
        model = create_model_ED_CNN(input_dim=w, units=units, b_size=batch_size, comp=comp)
    elif model_name == 'S4D':
        model = create_model_S4D(input_dim=w, units=units, b_size=batch_size, comp=comp)
    elif model_name == 'LSTM':
        model = create_model_LSTM(input_dim=w, b_size=batch_size, comp=comp)

    losses = 'mse'
    model.compile(loss=losses, optimizer=opt)
    start = time.time()
    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")


        loss_training = np.empty(epochs1)
        loss_val = np.empty(epochs1)
        best_loss = 1e9
        count = 0
        for i in range(epochs0, epochs1, 1):
            start = time.time()
            print('epochs:', i)
            model.reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=test_gen, callbacks=callbacks)
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                count = count + 1
                if count == 50:
                    break
            avg_time_epoch = (time.time() - start)

            sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch/60)} min")
            sys.stdout.write("\n")

        # save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs0)

        loss_training = np.array(loss_training)
        loss_val = np.array(loss_val)
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs0))

        print("Training done")
        
        
    avg_time_epoch = (time.time() - start)
    sys.stdout.write(f" Average time training{'{:.3f}'.format(avg_time_epoch/60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()

    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()

    # compute test loss
    model.reset_states()
    predictions = model.predict(test_gen, verbose=0)[:,-1]
    w = w - 1
    predictWaves(predictions, test_gen.x[w:len(predictions)+w], test_gen.y[w:len(predictions)+w], model_save_dir, save_folder, fs, '0')
     
    mse = tf.keras.metrics.mean_squared_error(test_gen.y[w:len(predictions)+w], predictions)
    mae = tf.keras.metrics.mean_absolute_error(test_gen.y[w:len(predictions)+w], predictions)
    esr = ESR(test_gen.y[w:len(predictions)+w], predictions)
    rmse = RMSE(test_gen.y[w:len(predictions)+w], predictions)
    #sftf_t =STFT_t(test_gen.y[w:len(predictions)+w], predictions)
    #sftf_f = STFT_f(test_gen.y[w:len(predictions)+w], predictions)
    #spectral_flux = flux(test_gen.y[w:len(predictions)+w], predictions, fs)
    lo = None#losses(test_gen.y[w:len(predictions)+w], predictions),
    #results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse, 'spectral_flux': spectral_flux, 'sftf_t': sftf_t, 'sftf_f': sftf_f}
    results_ = {'loss': lo, 'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse}

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(model_name) + 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)
            
    return 42
