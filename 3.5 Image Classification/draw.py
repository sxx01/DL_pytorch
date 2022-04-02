from pylab import *
import matplotlib.animation as animation
import random

class Monitor(object):
    """  This is supposed to be the class that will capture the data from
        whatever you are doing.
    """    
    def __init__(self,N):
        self._t    = linspace(0,100,N)
        self._data = self._t*0

    def captureNewDataPoint(self):
        """  The function that should be modified to capture the data
            according to your needs
        """ 
        return 2.0*rand()-1.0


    def updataData(self):
        while True:
            self._data[:]  = roll(self._data,-1)
            self._data[-1] = self.captureNewDataPoint()
            yield self._data

class StreamingDisplay(object):

    def __init__(self):
        self._fig = figure()
        self._ax  = self._fig.add_subplot(111)

    def set_labels(self,xlabel,ylabel):
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)

    def set_lims(self,xlim,ylim):
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

    def plot(self,monitor):
        self._line, = (self._ax.plot(monitor._t,monitor._data))

    def update(self,data):
        self._line.set_ydata(data)
        return self._line

# Main
if __name__ == '__main__':
    x = []
    y = []
    fig, ax = plt.subplots()
    ax.set_xlim([0, 11])
    ax.set_ylim([-1, 1])
    for i in range(10):
        ax = plt.subplot(111)
        plt.clf()
        x.append(i)
        y.append(random.random())
        ax.plot(x, y)
        plt.show()
        plt.pause(0.4)
    plt.ioff()  # 关闭画图的窗口，即关闭交互模式
    plt.show()  # 显示图片，防止闪退