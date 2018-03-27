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

    train_path = args.train
    predictions_path = args.predictions
    print train_path
    train_data = pd.read_csv(train_path)
    predictions_data = pd.read_csv(predictions_path)

    std = np.sqrt(predictions_data['variance'].values)

    fig = plt.figure(figsize=(8, 8))
    plt.fill_between(predictions_data['x'],
                     predictions_data['y'] - 3 * std,
                     predictions_data['y'] + 3 * std, color='steelblue', alpha=0.1,
                     label='+/- 3 sigma')
    plt.fill_between(predictions_data['x'],
                     predictions_data['y'] - std,
                     predictions_data['y'] + std, color='steelblue',
                     alpha=0.5, label='+/- sigma')
    plt.plot(predictions_data['x'],
             predictions_data['y'], color='steelblue',
             label='mean')
    plt.plot(predictions_data['x'],
             predictions_data['truth'].astype('float'), color='black',
             label='truth')
    plt.scatter(train_data['x'],
                train_data['y'], color='k',
                label='training points')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    fig.tight_layout()
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()
