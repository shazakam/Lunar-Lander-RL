from copy import deepcopy
import gymnasium as gym

import torch
from ddqn_agent import DDQNAgent
import os


def test_agent(
    agent,
    env,
    n_episodes=1,
    max_t=1000,
    eps=0.0,
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

    for _ in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        for _ in range(max_t):

            agent.online_network.eval()
            agent.target_network.eval()
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break


if __name__ == "__main__":

    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    opt_params = {
        "batch_size": 179,
        "fc1_units": 55,
        "fc2_units": 128,
        "include_optional_layer_1": False,
        "include_optional_layer_2": True,
        "fc4_units": 19,
        "gamma": 0.9999362438641551,
        "lr": 0.0025872698903683716,
        "eps_decay": 0.9266985052054595,
        "tau": 0.0007259979059427554,
        "loss_fn": "huber",
    }

    agent = DDQNAgent(
        seed=42,
        action_size=4,
        state_size=8,
        batch_size=opt_params["batch_size"],
        fc1_units=opt_params["fc1_units"],
        fc2_units=opt_params["fc2_units"],
        # fc3_units=opt_params['fc1_units'],
        fc4_units=opt_params["fc4_units"],
        gamma=opt_params["gamma"],
        lr=opt_params["lr"],
        eps_decay=opt_params["eps_decay"],
        tau=opt_params["tau"],
        loss_fn=opt_params["loss_fn"],
    )

    dirs = os.listdir("DDQN/saved_agents")

    for episode in dirs:

        record_env = gym.wrappers.RecordVideo(
            env,
            name_prefix=episode,
            video_folder=f"videos/ddqn",
        )
        networks = os.listdir(f"DDQN/saved_agents/{episode}")

        print(networks, episode)

        o_n = networks[0] if "online" in networks[0] else networks[1]
        t_n = networks[1] if "target" in networks[1] else networks[0]

        online_network = deepcopy(
            torch.load(
                f"DDQN/saved_agents/{episode}/{o_n}",
                map_location=torch.device("cpu"),
            )
        )
        target_network = deepcopy(
            torch.load(
                f"DDQN/saved_agents/{episode}/{t_n}",
                map_location=torch.device("cpu"),
            )
        )

        agent.target_network.load_state_dict(target_network)
        agent.online_network.load_state_dict(online_network)
        test_agent(agent=agent, env=record_env)
