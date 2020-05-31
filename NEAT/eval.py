import gym
import neat
import os
import pickle


def main(genomes, config):
    genomes = [genomes]
    nets = []
    ge = []
    agents = []

    for g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        env = gym.make('MountainCar-v0')
        env.reset()
        agents.append(env)
        g.fitness = 0
        ge.append(g)

    run = True

    while run:
        if len(agents) == 0:
            run = False
            break
        agents[0].render()
        dead = []
        for x, agent in enumerate(agents):
            output = nets[x].activate(agent.state)

            if output[0] > 0.5:
                action = 0
            elif output[1] > 0.5:
                action = 1
            else:
                action = 2

            _, r, done, _ = agent.step(action)
            ge[x].fitness += r
            if done:
                dead.append(x)

        for agent in reversed(dead):
            ge[agent].fitness -= 0.1
            print(ge[agent].fitness+0.1)
            nets.pop(agent)
            ge.pop(agent)
            agents[agent].close()
            agents.pop(agent)


if __name__ == '__main__':
    with open('best.pickle', 'rb') as f:
        winner = pickle.load(f)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)

    main(winner, config)

