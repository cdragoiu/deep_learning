#!/usr/local/bin/python3

from argparse import ArgumentParser
from termcolor import colored
from cnn_models.binary_image_classifier import BinaryImageClassifier

def run_model():

    # available models
    models = {
        BinaryImageClassifier.__name__: BinaryImageClassifier
    }

    # set up command-line options
    info = 'Run one of the following models: ' + colored(', '.join(models.keys()), 'green') + '.'
    parser = ArgumentParser(description=info, add_help=False)

    info = 'location of train/test data (proper folder structure required)'
    parser.add_argument('positional', metavar='path', nargs=1, help=info)

    required = parser.add_argument_group('required arguments')
    required.add_argument('-m', '--model', metavar='name', choices=models.keys(), required=True,
        help='select a model to run')

    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-t', '--train', action='store_true', help='train the model')
    optional.add_argument('-p', '--predict', action='store_true', help='perform predictions')
    optional.add_argument('-s', '--save', action='store_true', help='save model weights & history')
    optional.add_argument('-l', '--load', action='store_true', help='load model weights & history')
    optional.add_argument('-d', '--display', action='store_true', help='display model metrics')
    optional.add_argument('-h', '--help', action='help', help='show this help message and exit')
    args = parser.parse_args()

    # run the model
    model = models[args.model](args.positional[0])

    if args.load:
        model.load()

    if args.train:
        model.train()

    if args.save:
        model.save()

    if args.predict:
        model.predict()

    if args.display:
        model.display()

if __name__ == '__main__':
    run_model()
