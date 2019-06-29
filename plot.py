import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

class PlotMaker:

    def __init__(self, pa):
        self.pa = pa
        self.slow_down_records = []


    def plot(self):
        ax = plt.figure().gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        iterations = range(1, len(self.slow_down_records) + 1)
        colors = ['k','b','r','g','m','y','pink','darkred','c']
        self.slow_down_records = np.array(self.slow_down_records)
        for i in range(self.pa.num_job_priorities):
            priority_slice = self.slow_down_records[:,i]
            mean_slow_down = [np.mean(r) for r in priority_slice]
            plt.plot(iterations,mean_slow_down, colors[i], label='%.1f' % ((i + 1) * 0.1))
        plt.title('Average Slowdown Prioritized')
        plt.xlabel('iteration')
        plt.ylabel('average slowdown')
        plt.savefig('average_slowdown_prioritized.png')
        plt.clf()

        mean_slow_down = np.mean([np.concatenate(r) for r in self.slow_down_records],axis=1)
        plt.plot(iterations, mean_slow_down, 'k')
        plt.title('Average Slowdown')
        plt.xlabel('iteration')
        plt.ylabel('average slowdown')
        plt.savefig('average_slowdown.png')
        print("plotted!")
        plt.clf()

