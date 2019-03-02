""" Plotting of Neural Networks results """
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os


def mkdir(path):
    """Creates directory if it does not exist"""
    try:
        os.makedirs(path)
    except:
        pass


def plot_prediction(y, y_pred, index=""):
    """Plots the difference between actual and predicted price"""
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, color='red', alpha=0.3, linewidth=0.2, s=0.7)
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    #transform = ax.transAxes
    ax.add_line(line)
    plt.ylim(ymin=0, ymax=0.30)
    plt.xlim(xmin=0, xmax=0.30)
    plt.xlabel('Actual price')
    plt.ylabel('Predicted price')
    plt.show()
    mkdir_path = "img\\"+str(index)
    path = mkdir_path+"\\"+str(index)+"_prediction.png"
    mkdir(mkdir_path)
    fig.savefig(path, format="png", dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots()
    plt.hist(y-y_pred, range=((-0.06, 0.06)),
             bins=40, edgecolor='white', color='teal')
    plt.xlabel('Difference between actual and predicted price')
    plt.ylabel('Occurence')
    plt.show()
    mkdir_path = "img\\"+str(index)
    path = mkdir_path+"\\"+str(index)+"_prediction_histogram.png"
    mkdir(mkdir_path)
    fig.savefig(path, format="png", dpi=200, bbox_inches='tight')


def plot_errors(y, y_pred, index=""):
    """Plots MAE and MAPE errors between actual and predicted price"""
    fig, ax = plt.subplots()
    AE = abs(y-y_pred)
    plt.ylabel('Absolute Error')
    plt.xlabel('C/K')
    plt.xlim(xmin=0, xmax=0.5)
    plt.ylim(ymin=0, ymax=1)
    plt.scatter(y, AE, alpha=0.2, s=0.5, color='black')
    plt.show()
    mkdir_path = "img\\"+str(index)
    path = mkdir_path+"\\"+str(index)+"_absolute_error.png"
    mkdir(mkdir_path)
    fig.savefig(path, format="png", dpi=200, bbox_inches='tight')

    fig, ax = plt.subplots()
    APE = abs((y-y_pred)/y)
    plt.ylabel('Absolute Percent Error')
    plt.xlabel('Moneyness')
    plt.xlim(xmin=0, xmax=2)
    plt.ylim(ymin=0, ymax=1)
    plt.scatter(y, APE, alpha=1.0, marker='.', s=1, color='black')
    plt.show()
    mkdir_path = "img\\"+str(index)
    path = mkdir_path+"\\"+str(index)+"_absolute_percent_error.png"
    mkdir(mkdir_path)
    fig.savefig(path, format="png", dpi=200, bbox_inches='tight')
