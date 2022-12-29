# Lunar Lander V2

### DDQN Agent

To reproduce results for ddqn agent open ddqn_main.ipynb in google colab and run it from there. Further instructions are in the notebook. The auto hyperparameter tuning takes very long to run without a very good GPU/TPU so be careful. Also, take note to comment out lines that save files to drive or replace the save_location to your own drive.

To reproduce results from running the saved agent from training for 100, 200, 300, 400 and 500 episodes run ddqn_agent_video.py (this will create videos of the agent acting in the environment after being trained for the specified episodes). This was implemented in this way as google colab cannot render the Lunar Lander environment.s