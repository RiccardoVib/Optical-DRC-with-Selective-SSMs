import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves, MyLRScheduler
from Models import create_model_ED_baseline, create_model_LSTM_baseline
from DatasetsClass_baseline import DataGeneratorPicklesCL1B, DataGeneratorPicklesLA2A
import numpy as np
from Metrics import ESR, RMSE
import sys
import time
from Utils import has_numbers

def train(**kwargs):
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param save_folder: the directory in which the models are saved [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param comp: if LA2A or CL1B [string]
      :param model_name: type of the model [string]
      :param dataset: name of the datset to use [string]
      :param epochs: the number of epochs [int]
    """
    data_dir = kwargs.get('data_dir', '../../../Files/')
    save_folder = kwargs.get('save_folder', 'Testing')
    batch_size = kwargs.get('batch_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    units = kwargs.get('units', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    comp = kwargs.get('comp', None)
    model_name = kwargs.get('model', None)
    epochs = kwargs.get('epochs', 60)
    mini_batch_size = kwargs.get('mini_batch_size', 1)


    # set all the seed in case reproducibility is desired
    #np.random.seed(422)
    #tf.random.set_seed(422)
    #random.seed(422)

    # load the data generator according to the type of dataset
    if comp == 'CL1B':
        data_generator = DataGeneratorPicklesCL1B
    elif comp == 'LA2A':
        data_generator = DataGeneratorPicklesLA2A

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
        # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    fs = 48000
    w = 64
    
    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder, 0.)

    # to load the right test set
    dataset_test = dataset
    while has_numbers(dataset_test[8:]):
        dataset_test = dataset_test[:-1]

    # create the model
    if model_name == 'LSTMb':
        model = create_model_LSTM_baseline(input_dim=w, mini_batch_size=mini_batch_size, b_size=batch_size, comp=comp)
    elif model_name == 'EDb':
        model = create_model_ED_baseline(input_dim=w, mini_batch_size=mini_batch_size, units=units, b_size=batch_size, comp=comp)

    # create the DataGenerator object to retrieve the data
    train_gen = data_generator(data_dir, dataset + '_train.pickle', input_size=w, mini_batch_size=mini_batch_size,
                               batch_size=batch_size, model=model)
    test_gen = data_generator(data_dir, dataset_test + '_test.pickle', input_size=w, mini_batch_size=mini_batch_size,
                              batch_size=batch_size, model=model)

    # the number of total training steps
    training_steps = train_gen.training_steps*epochs
    # define the Adam optimizer with the initial learning rate, training steps
    opt = tf.keras.optimizers.Adam(learning_rate=MyLRScheduler(learning_rate, training_steps), clipnorm=1)
    


    # compile the model with the optimizer and selected loss function
    losses = 'mse'
    model.compile(loss=losses, optimizer=opt)

    # start the timer for all the training process
    start = time.time()
    # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        # load the weights of the last epoch, if any
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir_latest))
            model.load_weights(last)
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")


        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        # counting for early stopping
        count = 0
        for i in range(epochs):
            # start the timer for each epoch
            start = time.time()
            print('epochs:', i)
            # reset the model's states
            model.reset_states()
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=False, validation_data=test_gen, callbacks=callbacks)
            
            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
            # if not count is increased by one and if equal to 50 the training is stopped
            else:
                count = count + 1
                if count == 50:
                    break
                    
            avg_time_epoch = (time.time() - start)

            sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch/60)} min")
            sys.stdout.write("\n")


        # write and save results
        writeResults(results, units, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        # plot the training and validation loss for all the training
        loss_training = np.array(loss_training[:i])
        loss_val = np.array(loss_val[:i])
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

        print("Training done")
        
        
    avg_time_epoch = (time.time() - start)
    sys.stdout.write(f" Average time training{'{:.3f}'.format(avg_time_epoch/60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()
    
    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    # reset the states before predicting
    model.reset_states()
    predictions = model.predict(test_gen, verbose=0)[:, -1]

    predictions = predictions.reshape(test_gen.y.shape[0], -1)
    predictions = predictions[:, mini_batch_size - w:]
    y = test_gen.y[:, :len(predictions[0])]
    x = test_gen.x[:, :len(predictions[0])]

    predictions = predictions.reshape(-1)
    y = y.reshape(-1)
    x = x.reshape(-1)

    # plot and render the output audio file, together with the input and target
    predictWaves(predictions, x, y, model_save_dir, save_folder, fs, '0')

    # compute test loss
    mse = tf.keras.metrics.mean_squared_error(y, predictions)
    mae = tf.keras.metrics.mean_absolute_error(y, predictions)
    esr = ESR(y, predictions)
    rmse = RMSE(y, predictions)

    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse}

    # writhe and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, str(model_name) + 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)
            
    return 42
