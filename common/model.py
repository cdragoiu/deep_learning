import os
import pandas as pd
from termcolor import colored
from common.visualizations import MetricsView

class Model:
    ''' Base model class. All models inherit from this class. '''

    def __init__(self):
        '''
        path:    placeholder for input files location
        history: pandas DataFrame to store model training metrics per epoch
        model:   placeholder for model to train
        metrics: placeholder for model training metrics
        '''

        self.path = None
        self.history = pd.DataFrame()
        self.model = None
        self.metrics = []

    def check_dirs(self, dirs):
        '''
        Check if given directories exist.
        Args:
            dirs: list of directories to check (relative to set path)
        '''

        for dir in dirs:
            dir = self.path + dir
            if not os.path.isdir(dir):
                print(colored('error: "{}" not found', 'magenta').format(dir))
                exit()

    def save(self):
        ''' Save model weights and history using the provided path. '''

        self.check_dirs(['..'])

        self.model.save_weights(self.path + '../weights.h5')
        self.history.to_hdf(self.path + '../history.h5', key='history')

    def load(self):
        ''' Load model weights and history using the provided path. '''

        self.check_dirs(['..'])

        # load model weights
        file_name = self.path + '../weights.h5'
        if os.path.isfile(file_name):
            self.model.load_weights(file_name)
        else:
            print(colored('warning: "{}" not found', 'blue').format(file_name))

        # load training history
        file_name = self.path + '../history.h5'
        if os.path.isfile(file_name):
            self.history = pd.read_hdf(file_name)
        else:
            print(colored('warning: "{}" not found', 'blue').format(file_name))

    def display(self):
        ''' Display model evaluation metrics. '''

        if self.history.empty:
            print(colored('warning: nothing to display...empty history', 'blue'))
            return
        metrics_view = MetricsView(self.history)
        metrics = [m.name for m in self.metrics]
        metrics.insert(0, 'loss')
        metrics_view.display(metrics=metrics)
