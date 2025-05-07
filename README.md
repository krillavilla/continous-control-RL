[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Project 2: Continuous Control

### Introduction

For this project, you will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

### Installation

To set up the project environment:

1. Clone this repository
2. **Python Version**: This project requires Python 3.6 for compatibility with the Unity ML-Agents environment.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   If you encounter any issues during installation, please refer to the `troubleshooting_guide.md` file.

### Unity Environment Setup

1. Download the Unity environment for Option 1 (single agent) that matches your operating system:
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

2. Place the unzipped environment files in the `envs/` directory. For example:
   - Linux: `envs/Reacher_Linux/Reacher.x86_64`
   - Mac: `envs/Reacher.app`
   - Windows: `envs/Reacher_Windows_x86_64/Reacher.exe`

3. Make sure the environment file has execute permissions (Linux/Mac):
   ```bash
   chmod +x envs/Reacher_Linux/Reacher.x86_64
   ```

### Instructions

#### Verify Environment Setup

To verify that your environment is set up correctly:

```bash
python src/verify_env.py --env_path envs/Reacher_Linux/Reacher.x86_64
```

Adjust the path to match your operating system and environment location.

#### Training the Agent

To train the agent with default parameters:

```bash
python src/train.py --env envs/Reacher_Linux/Reacher.x86_64
```

For a quick test with fewer episodes:

```bash
python src/train.py --env envs/Reacher_Linux/Reacher.x86_64 --n_episodes 5 --max_t 200
```

The trained model will be saved to the `models/` directory.

#### Running the Notebook

To explore the project in a Jupyter notebook:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory and open `Continuous_Control.ipynb`

3. **Important**: The notebook uses relative paths (`../envs/Reacher_Linux/Reacher.x86_64`) to access the environment files. Make sure you're running the notebook from the `notebooks/` directory, not the project root.

4. Follow the instructions in the notebook to train and evaluate the agent

### Troubleshooting

If you encounter any issues, please refer to the `troubleshooting_guide.md` file for solutions to common problems. For specific issues:

- **Pandas Version Compatibility**: If you encounter issues with pandas installation on Python 3.6, see the troubleshooting_guide.md file for a detailed solution.
