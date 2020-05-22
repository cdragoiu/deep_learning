from tensorflow.keras.callbacks import Callback
from termcolor import colored

class StopTraining(Callback):
    ''' Callback to stop the training of the selected model. '''

    def __init__(self, metric, threshold):
        '''
        Args:
            metric:    metric to decide when to stop the training
            threshold: training is stopped if the metric is above the given threshold
        '''

        self.metric = metric
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs):
        ''' Called at the end of each epoch during the training process. '''

        try:
            if logs[self.metric] > self.threshold:
                print(colored('\ninfo: stopping training : {} > {}', 'green').format(
                    self.metric, self.threshold))
                self.model.stop_training = True
        except KeyError:
            print(colored('\nwarning: StopTraining callback failed...metric not present', 'blue'))
