from collections import deque
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler


from DDQN.ddqn_agent import DDQNAgent
from DQN.dqn_agent import DQNAgent


def train_agent(
    agent,
    env,
    n_episodes=1000,
    max_t=1000,
    eps_start=1.0,
    eps_end=0.01,
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
    scores = []
    episode_lengths = []
    losses = []
    exploitative_actions = []
    exploratory_actions = []

    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    eps_change = [eps]

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        episode_length = 0
        for _ in range(max_t):

            # Increment the episode length counter
            episode_length += 1

            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        # save total loss during the episode and reset it
        # losses.append(agent.loss.detach().numpy())
        agent.loss = 0

        exploitative_actions.append(agent.num_exploitative_actions)
        agent.num_exploitative_actions = 0

        exploratory_actions.append(agent.num_exploratory_actions)
        agent.num_exploratory_actions = 0

        episode_lengths.append(episode_length)

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score

        eps = max(eps_end, agent.eps_decay * eps)  # decrease epsilon
        eps_change.append(eps)
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

    return {
        "scores": scores,
        "episode_lengths": episode_lengths,
        "losses": losses,
        "exploitative_actions": exploitative_actions,
        "exploratory_actions": exploratory_actions,
        "eps_change": eps_change,
    }


def get_optimal_hyperparamters(env, agent_type, n_trials=10, n_episodes=1000, seed=42):
    def objective(trial):

        # Sample hyperparameter values
        buffer_size = trial.suggest_int("buffer_size", 1000, 10000)
        batch_size = trial.suggest_int("batch_size", 32, 256)
        fc1_units = trial.suggest_int("fc1_units", 16, 128)
        fc2_units = trial.suggest_int("fc2_units", 16, 128)
        gamma = trial.suggest_float("gamma", 0.9, 1.0)
        tau = trial.suggest_float("tau", 1e-5, 1e-3)
        lr = trial.suggest_float("lr", 1e-4, 1e-2)
        update_every = trial.suggest_int("update_every", 1, 6)
        eps_decay = trial.suggest_float("eps_decay", 0.9, 0.999)

        # Create and train DDQN agent
        agent = (
            DDQNAgent(
                state_size=8,
                action_size=4,
                seed=seed,
                buffer_size=buffer_size,
                batch_size=batch_size,
                fc1_units=fc1_units,
                fc2_units=fc2_units,
                eps_decay=eps_decay,
                gamma=gamma,
                tau=tau,
                lr=lr,
                update_every=update_every,
            )
            if agent_type == "ddqn"
            else DQNAgent(
                state_size=8,
                action_size=4,
                seed=seed,
                buffer_size=buffer_size,
                batch_size=batch_size,
                fc1_units=fc1_units,
                fc2_units=fc2_units,
                eps_decay=eps_decay,
                gamma=gamma,
                tau=tau,
                lr=lr,
                update_every=update_every,
            )
        )
        metrics = train_agent(agent, env, n_episodes=n_episodes)
        # Return average reward over all episodes
        return np.average(metrics["scores"])

    study = optuna.create_study(
        study_name=f"{agent_type}_study",
        direction="maximize",
        sampler=TPESampler(seed=seed),
        storage=None,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(
        f"hyperparams_opt/{agent_type}/{agent_type}_optimization_history.png",
        bbox_inches="tight",
    )

    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(
        f"hyperparams_opt/{agent_type}/{agent_type}_plot_slice.png",
        dpi=300,
        format="png",
    )

    importances = optuna.importance.get_param_importances(study)
    _save_bar_plot(
        data=importances,
        x_axis_name="Hyperparamters",
        y_axis_name="Importance",
        fig_name=f"hyperparams_opt/{agent_type}/{agent_type}_hyperparameter_importances.png",
    )
    return study.best_params


def _save_bar_plot(data, x_axis_name, y_axis_name, fig_name):
    values = [x[1] for x in data.items()]
    labels = [x[0] for x in data.items()]
    # Create the bar plot
    fig, ax = plt.subplots()

    colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet", "pink"]
    ax.bar(labels, values, color=colors)

    # Set the labels for the x-axis and y-axis
    ax.set_xlabel(x_axis_name)
    ax.set_ylabel(y_axis_name)

    # Rotate the labels for the x-axis
    plt.xticks(rotation=45)
    plt.savefig(fig_name, bbox_inches="tight")
    plt.clf()


def save_metric_plots(metrics, agent_type):
    title = lambda x: x.replace("_", " ").title()
    for key, metric in metrics.items():
        # clear plot so others don't get saved in same img
        plt.clf()
        key_title = title(key)
        plt.plot(metric)
        plt.xlabel("Episodes")
        plt.ylabel(key_title)
        plt.title(key_title + " over time")
        plt.savefig(f"metrics/{agent_type}/{agent_type}_{key}", bbox_inches="tight")
        # clear plot so others don't get saved in same img
        plt.clf()
