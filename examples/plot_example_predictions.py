import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("train")
    p.add_argument("predictions")
    p.add_argument("--output")
    return p


if __name__ == "__main__":

    p = create_parser()
    args = p.parse_args()

    # read in the training and prediction data
    train_path = args.train
    predictions_path = args.predictions
    print train_path
    train_data = pd.read_csv(train_path)
    x_name = 'feature'
    y_name = 'prediction'

    predictions_data = pd.read_csv(predictions_path)
    std = np.sqrt(predictions_data['prediction_variance'].values)

    fig = plt.figure(figsize=(8, 8))
    # create +/- 3 sigma shading
    plt.fill_between(predictions_data[x_name],
                     predictions_data[y_name] - 3 * std,
                     predictions_data[y_name] + 3 * std, color='steelblue', alpha=0.1,
                     label='+/- 3 sigma')
    # and +/- 1 sigma shading
    plt.fill_between(predictions_data[x_name],
                     predictions_data[y_name] - std,
                     predictions_data[y_name] + std, color='steelblue',
                     alpha=0.5, label='+/- sigma')
    # Plot the mean
    plt.plot(predictions_data[x_name],
             predictions_data[y_name], color='steelblue',
             label='mean')
    # Plot the truth
    plt.plot(predictions_data[x_name],
             predictions_data['target'].astype('float'), color='black',
             label='truth')
    # Show the training points
    plt.scatter(train_data['feature'],
                train_data['target'], color='k',
                label='training points')

    y_min = np.min(predictions_data['target'].astype('float'))
    y_max = np.max(predictions_data['target'].astype('float'))
    y_range = y_max - y_min
    plt.ylim([y_min - 0.1 * y_range, y_max + 0.1 * y_range])

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    fig.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
