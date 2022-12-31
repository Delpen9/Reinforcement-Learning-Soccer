import matplotlib.pyplot as plt

class Grapher():
    def __init__(self, x_axis, x_axis_label, y_axis, y_axis_label, title, ylim):
        self.x_axis = x_axis
        self.x_axis_label = x_axis_label
        self.y_axis = y_axis
        self.y_axis_label = y_axis_label
        self.title = title
        self.ylim = ylim

    def Graph(self):
        plt.title(self.title)
        plt.plot(self.y_axis, linestyle='-', linewidth=0.3)
        plt.xlabel(self.x_axis_label)
        plt.ylabel(self.y_axis_label)
        plt.ylim(0, self.ylim)
        plt.show()
