import gym
import numpy as np
from DDQN import Agent as DDQN
from DDPG import Agent as DDPG
from utils import plot_learning, grayscale
import hickle

env = gym.make("CarRacing-v2", domain_randomize=False, continuous=False, render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, 'video', episode_trigger=lambda x: x % 5 == 0)
agent = DDQN(input_dim=(96, 96, 1), n_actions=5, lr=0.001, gamma=0.99, epsilon_end=0.02)
# agent = DDPG(input_dim=(96, 96, 1), n_actions=3, lr1=0.001, lr2=0.002, gamma=0.99)

episodes = 500
scores = []
avg_scores = []

for episode in range(episodes):

    observation, info = env.reset()
    observation = grayscale(observation)

    play = True
    score = 0

    while play:
        action = agent.make_action(observation)

        observation_, reward, terminated, truncated, info = env.step(action)
        observation_ = grayscale(observation_)

        agent.store_data(observation, action, reward, observation_, terminated)
        agent.train(64)

        observation = observation_
        score += reward

        if terminated or truncated:
            play = False

    scores.append(score)
    avg_scores.append(np.mean(scores[max(0, episode - 100):(episode + 1)]))

    print('Episode:{} Score:{} AVG Score:{}'.format(episode+1, scores[episode], avg_scores[episode]))

    if (episode+1) % 100 == 0:
        agent.save_model(f'model/dqn_model_{episode+1}.h5')

agent.save_model('model/dqn_model_final.h5')
plot_learning(scores, avg_scores, 'Wyniki Agenta', 'plot/car_racing_dqn.png')
hickle.dump([scores, avg_scores], 'data/scores_dqn.hkl')

env.close()
