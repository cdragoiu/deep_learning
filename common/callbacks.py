from tensorflow.keras.callbacks import Callback

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
                print('\n\ninfo: stopping training : {} > {}\n'.format(self.metric, self.threshold))
                self.model.stop_training = True
        except KeyError:
            print('\n\nwarning: StopTraining callback failed...metric not present\n')
