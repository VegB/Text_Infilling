import numpy as np
import gym
from texar.agents import NatureDQNAgent

import tensorflow as tf

env = gym.make('CartPole-v1')

if __name__ == '__main__':
    hparams = NatureDQNAgent.default_hparams()
    hparams['qnetwork'] = {
        'hparams': {
            'network_hparams': {
                'layers': [
                    {
                        'type': 'Dense',
                        'kwargs': {
                            'units': 128,
                            'activation': 'relu'
                        }
                    }, {
                        'type': 'Dense',
                        'kwargs': {
                            'units': 128,
                            'activation': 'relu'
                        }
                    }, {
                        'type': 'Dense',
                        'kwargs': {
                            'units': 2
                        }
                    }
                ]
            }
        }
    }
    agent = NatureDQNAgent(actions=2, state_shape=(4, ), hparams=hparams)

    for i in range(5000):
        reward_sum = 0.0
        observation = env.reset()
        agent.set_initial_state(observation=observation)
        while True:
            action = agent.get_action()
            action_id = np.argmax(action)

            next_observation, reward, is_terminal, info = env.step(action=action_id)
            agent.perceive(next_observation=next_observation, action=action, reward=reward, is_terminal=is_terminal)

            reward_sum += reward
            if is_terminal:
                break
        print 'episode {round_id}: {reward}'.format(round_id=i, reward=reward_sum)