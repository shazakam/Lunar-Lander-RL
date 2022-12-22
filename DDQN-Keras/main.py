import gymnasium as gym

from ddqn import DDQN
import numpy as np
import logging

if __name__ == "__main__":
    logger = logging.getLogger("gym")
    logger.setLevel(logging.ERROR)
    env = gym.make("LunarLander-v2")
    agent = DDQN(
        alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0, batch_size=64, input_dims=8
    )

    n_games = 500
    scores = []
    epsilon_history = []

    # env = gym.wrappers.Monitor(
    #     env, "tmp/lunar_lander", video_callable=lambda ep_id: True, force=True
    # )

    for i in range(n_games):
        done = False
        score = 0
        observation, _ = env.reset(seed=42)

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            _ = agent.learn()

        epsilon_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100) : (i + 1)])
        print("episode", i, "score %.2f" % score, "average score %.2f" % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()
