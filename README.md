# Lunar Lander V2

### DQN First Implementation
To reproduce results from the First DQN Implementation run the DQN_tuner file using the tune_DQN function to run a training session with the desired hyperparameters. Results for this implementation will be exported to an Excel file in the Hyperparameter_Results folder in the DQN folder. Comment/Uncomment any lines for further results. The agent will run for 1000 training episodes by default.

Similarly, for the Modified DQN just ensure that your model.py and DQN agent are the ones found in the DQN_modified folder.

### DDQN Agent

To reproduce results for ddqn agent open ddqn_main.ipynb in google colab and run it from there. Further instructions are in the notebook. The auto hyperparameter tuning takes very long to run without a very good GPU/TPU so be careful. Also, take note to comment out lines that save files to drive or replace the save_location to your own drive.

To reproduce results from running the saved agent from training for 100, 200, 300, 400 and solved episodes run ddqn_agent_video.py in the DDQN folder (this will create videos of the agent acting in the environment after being trained for the specified episodes). This was implemented in this way as google colab cannot render the Lunar Lander environment.
