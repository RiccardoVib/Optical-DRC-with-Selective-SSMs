# Modeling Time-Variant Responses of Optical Compressors With Selective State Space Models


This code repository for the article _Modeling Time-Variant Responses of Optical Compression with Selective State Space Models, Journal of the Audio Engineering Society, 2025 March - Volume 73 Number 3.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our companion page with [Audio Samples](https://riccardovib.github.io/Optical-DRC-SSM_pages/)

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)
3. [VST Download](#vst-download)

<br/>

# Datasets

Datsets are available [here](https://www.kaggle.com/datasets/riccardosimionato/optical-dynamic-range-compressors-la-2a-cl-1b/versions/1)

Our architectures were evaluated on two optical compressors:
- Teletronix LA-2A optical compressor
- TubeTech CL 1B optical compressor

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./Code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use (LA2A, CL1B). [ [str] ] (default=[" "] )
* --comp - The names of the device to consider (LA2A, CL1B). [ [str] ] (default=[" "] )
* --epochs - Number of training epochs. [int] (default =60)
* --model - The name of the model to train ('LSTM', 'ED', 'LRU', 'S4D', 'S6') [str] (default=" ")
* --batch_size - The size of each batch [int] (default=8 )
* --units = The hidden layer size (amount of units) of the network. [ [int] ] (default=8)
* --mini_batch_size - The mini batch size [int] (default=2048)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)
 

Example training case: 
```
cd ./Code/

python starter.py --datasets LA2A --comp LA2A --model LSTM --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --datasets LA2A --comp LA2A --model LSTM --only_inference True
```


# VST Download

Coming soon...