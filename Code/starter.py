import argparse
from Training import train

"""
Main starter script for training and inference.
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Training and inference the models.')

    parser.add_argument('--model_save_dir', default='./models', type=str, nargs='?', help='Folder directory in which to store the trained models.')

    parser.add_argument('--data_dir', default='./datasets', type=str, nargs='?', help='Folder directory in which the datasets are stored.')

    parser.add_argument('--datasets', default=[" "], nargs='+', help='The names of the datasets to use. Datasets = [LA2A, CL1B].')

    parser.add_argument('--comp', default=[" "], nargs='+', help='The names of the device to consider. [LA2A, CL1B].')

    parser.add_argument('--epochs', default=60, type=int, nargs='?', help='Number of training epochs.')

    parser.add_argument('--model', default=[" "], type=int, nargs='?', help='The name of the model to train (LSTM, ED, LRU, S4D, S6).')

    parser.add_argument('--batch_size', default=8, type=int, nargs='?', help='Batch size.')

    parser.add_argument('--mini_batch_size', default=2048, type=int, nargs='?', help='Mini batch size.')

    parser.add_argument('--units', default=8, nargs='+', help='Hidden layer sizes (amount of units) of the network.')

    parser.add_argument('--learning_rate', default=3e-4, type=float, nargs='?', help='Initial learning rate.')

    parser.add_argument('--only_inference', default=False, type=bool, nargs='?', help='When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model.')

    return parser.parse_args()


def start_train(args):

    print("######### Preparing for training/inference #########")
    print("\n")
    train(data_dir=args.data_dir,
          model_save_dir=args.model_save_dir,
          save_folder=f'{args.model}_{args.dataset}_{args.units}',
          dataset=args.datasets,
          comp=args.comp,
          epochs=args.epochs,
          model=args.model,
          batch_size=args.batch_size,
          mini_batch_size=args.mini_batch_size,
          units=args.units,
          learning_rate=args.learning_rate,
          inference=args.only_inference)


def main():
    args = parse_args()
    start_train(args)

if __name__ == '__main__':
    main()



