#!/usr/local/bin/python3

import os, random, shutil
from argparse import ArgumentParser
from termcolor import colored

def train_test_split(path, ratio):
    '''
    Split files into train and test based on the split ratio. The selection is randomized.
    Args:
        path:  location of files to process (subfolders are considered image classes)
        ratio: percentage of files to be used for training
    '''

    abs_path = os.path.abspath(path)
    dir_names = ['train', 'test']

    # reset folder structure if present
    for dir in dir_names:
        test_dir = abs_path + '/' + dir
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # process subfolders
    for class_name in os.listdir(abs_path):

        # skip hidden folders
        if class_name.startswith('.'):
            continue

        # create folder structure
        try:
            for dir in dir_names:
                os.makedirs(abs_path + '/' + dir + '/' + class_name)
        except OSError:
            print(colored('error: unable to create folder structure', 'magenta'))

        # randomize file selection
        split_files = {}
        file_names = os.listdir(abs_path + '/' + class_name)
        split_files[dir_names[0]] = random.sample(file_names, int(ratio * len(file_names)))
        split_files[dir_names[1]] = [f for f in file_names if f not in split_files[dir_names[0]]]

        # process files
        for dir in dir_names:
            for name in split_files[dir]:
                test_name = abs_path + '/' + class_name + '/' + name

                # skip empty files
                if os.path.getsize(test_name) == 0:
                    print(colored('warning: file "{}" is empty', 'blue').format(name))
                    continue

                os.symlink(test_name, test_name.replace(abs_path, abs_path + '/' + dir))

if __name__ == '__main__':

    # set up command-line options
    parser = ArgumentParser(
        description='Split files into train and test based on the split ratio.',
        add_help=False
    )
    parser.add_argument('positional', metavar='path', nargs=1,
        help='location of files to process (subfolders are considered image classes)'
    )
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--ratio', metavar='value', type=float, required=True,
        help='split ratio'
    )
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument('-h', '--help', action='help', help='show this help message and exit')
    args = parser.parse_args()

    # split files
    train_test_split(args.positional[0], args.ratio)
