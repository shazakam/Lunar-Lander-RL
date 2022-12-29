from DDQN.ddqn_agent import DDQNAgent


if __name__ == "__main__":
    agent = DDQNAgent(
        action_size=4,
        state_size=8,
        batch_size=180,
        fc1_units=58,
        fc2_units=36,
        gamma=0.9741438536769185,
        lr=0.008099294045021133,
        loss_fn="huber",
    )
