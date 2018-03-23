import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

if __name__ == "__main__":
    train_path = sys.argv[1]
    predictions_path = sys.argv[2]
    train_data = pd.read_csv(train_path)
    predictions_data = pd.read_csv(predictions_path)

    std = np.sqrt(predictions_data['variance'].values)

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
    plt.show()
