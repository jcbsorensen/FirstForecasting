# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_years(dataframe, feature, years=[]):
    if len(years) == 0:
        years = dataframe.index.year.unique().values.tolist()
    plt.figure()
    for i in range(len(years)):
        # prepare subplot
        ax = plt.subplot(len(years), 1, i + 1)
        # determine the year to plot
        year = years[i]
        # get all observations for the year
        result = dataframe[str(year)]
        # plot the active power for the year
        plt.plot(result[feature])
        # add a title to the subplot
        plt.title(str(year), y=0, loc='left')
        # turn off ticks to remove clutter
        plt.yticks([])
        plt.xticks([])
    plt.show()

def plot_months(dataframe, feature, year=None, start_month=1, end_month=12):
    months = [x for x in range(start_month, end_month+1)]
    if year is None:
        year = dataframe.index.year.unique().values.tolist()[0]
    plt.figure()
    for i in range(len(months)):
        # prepare subplot
        ax = plt.subplot(len(months), 1, i+1)
        # determine the month to plot
        month = str(year) + '-' + str(months[i])
        # get all observations for the month
        result = dataframe[month]
        # plot the active power for the month
        plt.plot(result[feature])
        # add a title to the subplot
        plt.title(month, y=0, loc='left')
        # turn off ticks to remove clutter
        plt.yticks([])
        plt.xticks([])
    plt.show()

