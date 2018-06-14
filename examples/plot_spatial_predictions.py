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
    predictions_data = pd.read_csv(predictions_path)
    std = np.sqrt(predictions_data['variance'].values)

    fig = plt.figure(figsize=(8, 8))
    n = int(np.sqrt(predictions_data.shape[0]))
    truth = predictions_data['prediction'].values.reshape(n, n)
    truth_isnan = np.isnan(truth).sum()
    truth[truth_isnan] = 0.
    plt.pcolormesh(predictions_data['x'].values.reshape(n, n),
                   predictions_data['y'].values.reshape(n, n),
                   truth,
                   vmin=-1., vmax=1.,
                cmap='coolwarm',
             label='truth')

    # Plot the mean
    plt.scatter(train_data['x'],
                train_data['y'],
                c=train_data['target'],
                cmap='coolwarm',
                   vmin=-1., vmax=1.,
             label='observations')

    plt.show()
