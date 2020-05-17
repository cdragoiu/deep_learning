import math
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

class MetricsView():
    ''' Visualization of model evalution metrics. '''

    def __init__(self, history):
        '''
        Args:
            history: pandas DataFrame containing model training metrics per epoch
        '''
        self.history = history

    def display(self, metrics):
        ''' Display given model evaluation metrics. '''

        # return if no metric is passed
        metrics_size = len(metrics)
        if metrics_size == 0:
            return

        # determine display setup
        if metrics_size < 4:
            columns = metrics_size
            rows = 1
        elif metrics_size == 4:
            columns = 2
            rows = 2
        else:
            columns = 3
            rows = math.ceil(metrics_size / columns)

        # set up the figure
        fig_width = 4 * columns
        fig_height = 2.4 * rows
        figure = plt.figure(figsize=(fig_width,fig_height))
        figure.subplotpars.update(
            left=0.6/fig_width,
            bottom=0.4/fig_height,
            right=1-0.6/fig_width,
            top=1-0.4/fig_height,
            wspace=0.18,
            hspace=0.36
        )
        axes = figure.subplots(rows, columns)
        if metrics_size == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # display metrics
        train_plot = None
        test_plot = None
        for i, name in enumerate(metrics):
            axes[i].set_title(name)
            axes[i].xaxis.set_major_locator(tkr.MaxNLocator(integer=True))
            if name in self.history.columns:
                epochs = self.history.index + 1
                train_plot, = axes[i].plot(epochs, self.history[name], color='#3399ff')
                val_name = 'val_' + name
                if val_name in self.history.columns:
                    test_plot, = axes[i].plot(epochs, self.history[val_name], color='#ff9933')

        # set up the legend
        legend_handles = [h for h in [train_plot, test_plot] if h is not None]
        legend_labels = ['train', 'test']
        figure.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc=(1-0.6/fig_width,1-0.8/fig_height),
            handlelength=1,
            handletextpad=0.4,
            frameon=False
        )

        # hide unused subplots
        for i in range(metrics_size, rows * columns):
            axes[i].set_visible(False)

        plt.show()
