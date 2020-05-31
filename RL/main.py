from RL.interface import Interact
import matplotlib.pyplot as plt
import numpy as np


def main():
    game = Interact()
    game.agent.load_model()
    game.train(200)
    r = game.scores
    plt.plot(np.arange(len(r)), r)
    plt.show()
    plt.savefig('test.png')
    # df.to_csv('Scores.csv', index=False)


main()


