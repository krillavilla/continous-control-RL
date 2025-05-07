# Project Report: Continuous Control

## Learning Algorithm

For this project, I implemented the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the Reacher environment (Option 1 - single agent). DDPG is an actor-critic, model-free algorithm that can operate in continuous action spaces.

### Algorithm Details

DDPG combines the following key components:

1. **Actor-Critic Architecture**:
   - **Actor Network**: Determines the best action for a given state
   - **Critic Network**: Evaluates the action by estimating the Q-value

2. **Experience Replay**: Stores and randomly samples experiences to break correlations between consecutive samples

3. **Target Networks**: Separate networks for stable learning that are updated softly

4. **Ornstein-Uhlenbeck Process**: Adds temporally correlated noise for better exploration in continuous action spaces

### Network Architectures

#### Actor Network
- Input Layer: State size (33)
- Hidden Layer 1: 400 units with ReLU activation and batch normalization
- Hidden Layer 2: 300 units with ReLU activation
- Output Layer: Action size (4) with tanh activation to bound actions between -1 and 1

#### Critic Network
- Input Layer: State size (33)
- Hidden Layer 1: 400 units with ReLU activation and batch normalization
- Hidden Layer 2: 300 units with ReLU activation (concatenated with actions after the first hidden layer)
- Output Layer: 1 unit (Q-value)

### Hyperparameters

The following hyperparameters were used for training:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Buffer Size | 1,000,000 | Replay buffer size |
| Batch Size | 128 | Minibatch size |
| Gamma | 0.99 | Discount factor |
| Tau | 0.001 | Soft update parameter for target networks |
| Learning Rate (Actor) | 0.0001 | Learning rate for the actor network |
| Learning Rate (Critic) | 0.001 | Learning rate for the critic network |
| Weight Decay | 0 | L2 weight decay for the critic |
| Network Architecture | 400, 300 | Units in first and second hidden layers |
| OU Noise Mu | 0.0 | Mean of the Ornstein-Uhlenbeck noise |
| OU Noise Theta | 0.15 | Parameter controlling the speed of mean reversion |
| OU Noise Sigma | 0.2 | Initial standard deviation of the noise |
| OU Noise Min Sigma | 0.05 | Minimum standard deviation (sigma will decay to this) |
| OU Noise Decay Period | 10000 | Number of steps over which to decay sigma |
| Noise Scale | 1.0 | Initial scale of the noise |
| Noise Decay | 0.995 | Decay rate for the noise scale |

### Hyperparameter Tuning

To achieve the required score of +30 over 100 consecutive episodes, extensive hyperparameter tuning was performed. The implementation now supports configurable hyperparameters through command-line arguments, allowing for systematic experimentation with different settings.

Key findings from the hyperparameter tuning process:

1. **Network Architecture**: Larger networks (400, 300 units) performed better than smaller ones (256, 128 units), likely due to the complexity of the continuous control task.

2. **Learning Rates**: A smaller learning rate for the actor (0.0001) and a larger one for the critic (0.001) provided the best balance between stability and learning speed.

3. **Exploration Noise**: The Ornstein-Uhlenbeck noise parameters significantly impact exploration efficiency. Adding a decay mechanism to gradually reduce exploration over time improved performance.

4. **Batch Size**: Increasing the batch size from 64 to 128 led to more stable learning and better final performance.

5. **Training Duration**: While some configurations solved the environment in around 150 episodes, others required up to 500-1000 episodes. Extended training capabilities were added to support longer training sessions when needed.

## Training Results

The agent was able to solve the environment in approximately 150 episodes. The environment is considered solved when the average (over 100 episodes) of those average scores is at least +30. The training curve below shows the score progression over episodes:

![Training Curve](models/scores.png)

The agent's performance improved steadily over time, with some fluctuations due to the exploration noise. After solving the environment, the agent continued to maintain high performance, demonstrating the stability of the learned policy.

## Extended Training Capabilities

To ensure the agent can reliably achieve the required score of +30 over 100 consecutive episodes, several extended training capabilities were implemented:

1. **Longer Training Sessions**: The implementation now supports training for up to 2000 episodes or more, with the ability to run in no-graphics mode for faster training.

2. **Configuration Management**: Different hyperparameter configurations can be named and saved separately, allowing for systematic comparison of different settings.

3. **Early Stopping**: Training automatically stops when the environment is solved (average score ≥ 30 over 100 episodes), saving time and computational resources.

4. **Performance Tracking**: Detailed logging and visualization of training progress help identify when the agent is improving and when it might be struggling.

5. **Headless Training**: Support for running the environment without visualization (`--no-graphics` flag) significantly speeds up training, especially for extended sessions.

These capabilities make it easier to experiment with different hyperparameters and training durations, ultimately leading to a more robust solution.

## Ideas for Future Work

Several improvements could be made to further enhance the agent's performance:

1. **Prioritized Experience Replay**: Instead of uniform sampling from the replay buffer, prioritize experiences based on their TD error magnitude. This would focus learning on the most informative transitions.

2. **Parameter Noise for Exploration**: Replace the Ornstein-Uhlenbeck process with parameter space noise for exploration, which can provide more consistent exploration throughout training.

3. **Distributional RL**: Implement a distributional critic that predicts the distribution of Q-values rather than just the expected value, which can lead to more robust learning.

4. **Multi-Agent Training**: Extend the implementation to Option 2 (20 agents) and leverage the parallel nature of the environment for faster and more stable learning.

5. **Advanced Hyperparameter Tuning**: Implement automated hyperparameter optimization using techniques like grid search, random search, or Bayesian optimization.

6. **Alternative Algorithms**: Implement and compare other continuous control algorithms such as Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), or Soft Actor-Critic (SAC).

7. **Network Architecture Improvements**: Experiment with different network architectures, such as deeper networks or residual connections, to potentially improve learning efficiency.

By implementing these improvements, the agent could potentially learn faster, achieve higher scores, and generalize better to different environments.
