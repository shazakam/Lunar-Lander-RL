import gymnasium as gym
from DDQN.ddqn_agent import DDQNAgent
from DQN.dqn_agent import DQNAgent
from utils import train_agent, plot_scores


if __name__ == "__main__":
    # Set the hyperparameters
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR = 5e-4  # learning rate
    UPDATE_EVERY = 4  # how often to update the network

    ddqn_agent = DDQNAgent(
        state_size=8,
        action_size=4,
        seed=42,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        lr=LR,
        update_every=UPDATE_EVERY,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
    )
    dqn_agent = DQNAgent(state_size=8, action_size=4, seed=0)

    env = gym.make("LunarLander-v2")
    scores = train_agent(dqn_agent, env, n_episodes=100)
    plot_scores(scores)
