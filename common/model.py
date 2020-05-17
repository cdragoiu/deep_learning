import pandas as pd
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

        self.path = ''
        self.history = pd.DataFrame()
        self.model = None
        self.metrics = []

    def save(self):
        ''' Save model weights and history using the provided path. '''

        self.model.save_weights(self.path + '/../weights.h5')
        self.history.to_hdf(self.path + '/../history.h5', key='history')

    def load(self):
        ''' Load model weights and history using the provided path. '''

        self.model.load_weights(self.path + '/../weights.h5')
        self.history = pd.read_hdf(self.path + '/../history.h5')

    def display(self):
        ''' Display model evaluation metrics. '''

        metrics_view = MetricsView(self.history)
        metrics = [m.name for m in self.metrics]
        metrics.insert(0, 'loss')
        metrics_view.display(metrics=metrics)
