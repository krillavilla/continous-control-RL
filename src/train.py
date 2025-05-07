import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from unityagents import UnityEnvironment

from agent import Agent
from utils import ReplayBuffer, OUNoise, plot_scores  # Added plot_scores import here

def train(env_path, n_episodes=1000, max_t=1000, print_every=10, 
          solve_score=30.0, window_size=100, save_dir='models', no_graphics=False):
    """Train the agent using DDPG.

    Params
    ======
        env_path (string): path to the Unity environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): print progress every N episodes
        solve_score (float): environment is considered solved when average score >= solve_score
        window_size (int): window size for calculating average score
        save_dir (string): directory to save model checkpoints
    """
    # Create the environment with an improved retry mechanism for different worker_ids
    print(f"Attempting to connect to environment at: {env_path}")
    max_retries = 10  # Increased from 5 to 10
    env = None

    # Try a wider range of worker_ids to avoid port conflicts
    for worker_id in range(10, 30):  # Use higher worker_ids (10-29) to avoid common port conflicts
        try:
            print(f"Attempting to connect with worker_id={worker_id}...")
            # Use only supported parameters
            env = UnityEnvironment(
                file_name=env_path, 
                worker_id=worker_id, 
                no_graphics=no_graphics
            )
            print(f"Successfully connected with worker_id={worker_id}")
            break
        except OSError as e:
            error_str = str(e)
            if "handle is closed" in error_str and worker_id < 29:
                print(f"Worker ID {worker_id} failed with 'handle is closed' error. Trying next worker_id...")
            elif "address already in use" in error_str.lower() and worker_id < 29:
                print(f"Worker ID {worker_id} failed with 'address already in use' error. Trying next worker_id...")
            else:
                print(f"Connection failed with error: {error_str}")
                if worker_id < 29:
                    print("Trying next worker_id...")
                else:
                    raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            if worker_id < 29:
                print("Trying next worker_id...")
            else:
                raise

    if env is None:
        raise Exception("Failed to connect to the Unity environment after multiple attempts.")

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Get environment information
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    print(f"Number of agents: {num_agents}")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")

    # Create agent
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize scores list and moving average
    scores = []
    scores_window = deque(maxlen=window_size)

    # Training loop
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = 0

        # Episode loop
        for t in range(max_t):
            # Select actions
            actions = agent.act(states)

            # Take actions in environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            # Update agent
            agent.step(states, actions, rewards, next_states, dones)

            # Update state and score
            states = next_states
            score += rewards[0]  # For single agent (Option 1)

            # Exit episode if done
            if dones[0]:
                break

        # Save score and update moving average
        scores_window.append(score)
        scores.append(score)

        # Print progress
        if i_episode % print_every == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        # Check if environment is solved
        if np.mean(scores_window) >= solve_score:
            print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}")

            # Save final model
            checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
            agent.save(checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

            # Plot scores
            fig = plot_scores(scores, window_size)
            plt.savefig(os.path.join(save_dir, 'scores.png'))

            break

    # Save final model regardless of whether environment is solved
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    agent.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Save scores to pickle file
    import pickle
    scores_path = os.path.join(save_dir, 'scores.pkl')
    with open(scores_path, 'wb') as f:
        pickle.dump(scores, f)
    print(f"Scores saved to {scores_path}")

    # Close environment
    env.close()

    return scores

def main():
    """Parse arguments and train the agent."""
    parser = argparse.ArgumentParser(description='Train a DDPG agent for continuous control')
    parser.add_argument('--env', type=str, required=True, help='path to Unity environment')
    parser.add_argument('--n_episodes', type=int, default=1000, help='maximum number of training episodes')
    parser.add_argument('--max_t', type=int, default=1000, help='maximum number of timesteps per episode')
    parser.add_argument('--print_every', type=int, default=10, help='print progress every N episodes')
    parser.add_argument('--solve_score', type=float, default=30.0, help='environment is considered solved when average score >= solve_score')
    parser.add_argument('--window_size', type=int, default=100, help='window size for calculating average score')
    parser.add_argument('--save_dir', type=str, default='models', help='directory to save model checkpoints')
    parser.add_argument('--no-graphics', action='store_true', help='Run the environment in no-graphics mode')

    args = parser.parse_args()

    # Train the agent
    scores = train(
        env_path=args.env,
        n_episodes=args.n_episodes,
        max_t=args.max_t,
        print_every=args.print_every,
        solve_score=args.solve_score,
        window_size=args.window_size,
        save_dir=args.save_dir,
        no_graphics=args.no_graphics
    )

    # Plot scores
    fig = plot_scores(scores, args.window_size)
    plt.savefig(os.path.join(args.save_dir, 'scores.png'))

if __name__ == '__main__':
    main()

def plot_scores(scores, window_size=100):
    """Plot scores and average scores over time."""
    # Create a larger figure (width, height in inches)
    fig = plt.figure(figsize=(12, 8))  # Increased from default size

    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)

    # Plot rolling average
    rolling_mean = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    plt.plot(np.arange(len(rolling_mean)) + window_size - 1, rolling_mean)

    # Add horizontal line at 30.0
    plt.axhline(y=30.0, color='r', linestyle='-')

    plt.ylabel('Score', fontsize=14)  # Larger font sizes
    plt.xlabel('Episode #', fontsize=14)
    plt.title('Training Progress', fontsize=16)
    plt.legend(['Score', f'Average Score (window={window_size})', 'Target Score (30.0)'], 
               fontsize=12)

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make sure everything fits
    plt.tight_layout()

    return fig
