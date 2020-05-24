from interface import Interact
import matplotlib.pyplot as plt
import numpy as np


def main():
    game = Interact()
    game.agent.load_model()
    r = game.observe(5)
    # r = game.scores
    plt.plot(np.arange(len(r)), r)
    plt.show()


main()


