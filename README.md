# Modeling Time-Variant Responses of Optical Compressors With Selective State Space Models


This code repository for the article _Modeling Time-Variant Responses of Optical Compression with Selective State Space Models, Journal of the Audio Engineering Society, 2025 March - Volume 73 Number 3.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our companion page with [Audio Samples](https://riccardovib.github.io/Optical-DRC-SSM_pages/)

### Folder Structure

```
./src
├── Code
└── Weights
    ├── CL1B_analog
    │   ├── ED-CNNCL1B_analog
    │   ├── EDbaselineCL1B_analog
    │   ├── LSTMbaselineCL1B_analog
    │   ├── LSTMCL1B_analog
    │   ├── MambaCL1B_analog
    │   └── S4DCL1B_analog
    └── LA2A_analog
        ├── ED-CNNLA2A_analog
        ├── EDbaselineLA2A_analog
        ├── LSTMbaselineLA2A_analog
        ├── LSTMLA2A_analog
        ├── MambaLA2A_analog
        └── S4DLA2A_analog
```

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

[VSTs](https://github.com/RiccardoVib/NeuralModelsVST/tree/main)

# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

```
@article{simionato2025modeling,
  title={Modeling Time-Variant Responses of Optical Compressors with Selective State Space Models},
  author={Simionato, Riccardo and Fasciani, Stefano},
  journal={Journal of Audio Engineering Society},
  volume={73},
  number={3},
  pages={144–165},
  year={2025}
}
```
