import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from Models import create_model_ED_original, create_model_LSTM_baseline
from DatasetsClass_baseline import DataGeneratorPicklesCL1B, DataGeneratorPicklesLA2A
import numpy as np
import random
from Metrics import ESR, RMSE
import sys
import time
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
    model_name = kwargs.get('model', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', 60)


    start = time.time()

    #####seed
    #np.random.seed(42)
    #tf.random.set_seed(42)
    #random.seed(42)

    if comp == 'CL1B':
        data_generator = DataGeneratorPicklesCL1B
    elif comp == 'LA2A':
        data_generator = DataGeneratorPicklesLA2A

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
        # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    fs = 48000
    w = 64
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, 0.)
    #
    dataset_test = dataset
    while has_numbers(dataset_test[8:]):
        dataset_test = dataset_test[:-1]

    # load the datasets
    train_gen = data_generator(data_dir, dataset + '_train.pickle', input_size=w, batch_size=batch_size)
    test_gen = data_generator(data_dir, dataset_test + '_test.pickle', input_size=w, batch_size=batch_size)

    training_steps = train_gen.training_steps*epochs
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)
    
    # create the model
    if model_name == 'LSTMbaseline':
        model = create_model_LSTM_baseline(input_dim=w, b_size=batch_size, comp=comp)
    elif model_name == 'EDbaseline':
        model = create_model_ED_original(input_dim=w, units=units, b_size=batch_size, comp=comp)


    losses = 'mse'
    model.compile(loss=losses, optimizer=opt)

    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")


        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        count = 0
        for i in range(epochs):
            
            print('epochs:', i)
            model.reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=test_gen, callbacks=callbacks)
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]

            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            else:
                count = count + 1
                if count == 30:
                    break
            avg_time_epoch = (time.time() - start)

            sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch/60)} min")
            sys.stdout.write("\n")

        # save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        loss_training = np.array(loss_training[:i])
        loss_val = np.array(loss_val[:i])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

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
    w = w-1
    predictWaves(predictions, test_gen.x[w:len(predictions)+w], test_gen.y[w:len(predictions)+w], model_save_dir, save_folder, fs, '0')
     
    mse = tf.keras.metrics.mean_squared_error(test_gen.y[w:len(predictions)+w], predictions)
    mae = tf.keras.metrics.mean_absolute_error(test_gen.y[w:len(predictions)+w], predictions)
    esr = ESR(test_gen.y[w:len(predictions)+w], predictions)
    rmse = RMSE(test_gen.y[w:len(predictions)+w], predictions)
    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse}

    with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(model_name) + 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)
            
    return 42