from interface import Interact
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    game = Interact()
    game.train(1000)
    r = game.scores
    plt.plot(np.arange(len(r)), r)
    plt.show()
    plt.savefig('Training.png')
    df = pd.Series(r)
    df.to_csv('Scores.csv', index=False)


main()


