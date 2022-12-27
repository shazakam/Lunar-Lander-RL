import gymnasium as gym
from DDQN.ddqn_agent import DDQNAgent
from DQN.dqn_agent import DQNAgent
from utils import (
    get_optimal_hyperparamters,
    save_metric_plots,
    train_agent,
)


def run_optimal_agent(
    agent_type,
    seed=42,
    n_episodes=1000,
    n_trials=10,
    env=gym.make("LunarLander-v2", render_mode="rgb_array_list"),
):
    env = env

    opt_params = get_optimal_hyperparamters(
        env, n_trials=n_trials, n_episodes=n_episodes, agent_type=agent_type
    )

    agent = (
        DDQNAgent(
            state_size=8,
            action_size=4,
            seed=seed,
            buffer_size=opt_params["buffer_size"],
            batch_size=opt_params["batch_size"],
            lr=opt_params["lr"],
            update_every=opt_params["update_every"],
            eps_decay=opt_params["eps_decay"],
            loss_fn=opt_params["loss_fn"],
        )
        if agent_type == "ddqn"
        else DQNAgent(
            state_size=8,
            action_size=4,
            seed=seed,
            buffer_size=opt_params["buffer_size"],
            batch_size=opt_params["batch_size"],
            lr=opt_params["lr"],
            update_every=opt_params["update_every"],
            eps_decay=opt_params["eps_decay"],
            loss_fn=opt_params["loss_fn"],
        )
    )

    # create new env with same seed and set render mode here as this makes training slower
    # when finding optimal hyperparameters
    record_env = gym.wrappers.RecordVideo(
        env,
        f"videos/{agent_type}",
        episode_trigger=lambda x: x % 100 == 0,
    )

    optim_agent_metrics = train_agent(
        agent, record_env, n_episodes=n_episodes, save_agent=True, agent_type=agent_type
    )
    save_metric_plots(optim_agent_metrics, agent_type=agent_type)


if __name__ == "__main__":
    # final inputs for real results should be: n_episodes=500, n_trials=10
    run_optimal_agent(agent_type="ddqn", n_episodes=1000, n_trials=10)

    # keep this commented out until DQN is adapated to work with this function
    # run_optimal_agent(agent_type="dqn", n_episodes=100, n_trials=2)
