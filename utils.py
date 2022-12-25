from collections import deque
import numpy as np
import matplotlib.pyplot as plt


def train_agent(
    agent,
    env,
    n_episodes=2000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            "\rEpisode {}\tAverage Score: {:.2f}".format(
                i_episode, np.mean(scores_window)
            ),
            end="",
        )
        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_window)
                )
            )
        if np.mean(scores_window) >= 200.0:
            print(
                "\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                    i_episode - 100, np.mean(scores_window)
                )
            )
            # torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break
    return scores


def plot_scores(scores1, scores2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot scores1 in subplot 1
    ax1.plot(scores1)
    ax1.set_title("DDQN")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score")

    # Plot scores2 in subplot 2
    ax2.plot(scores2)
    ax2.set_title("DQN")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Score")

    plt.show()
