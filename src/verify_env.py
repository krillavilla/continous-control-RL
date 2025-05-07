from unityagents import UnityEnvironment
import signal
import sys
import argparse


def signal_handler(sig, frame):
    """Handle cleanup when interrupting the program"""
    print("\nClosing environment gracefully...")
    try:
        if 'env' in globals():
            env.close()
    except Exception as e:
        print(f"Error during cleanup: {e}")
    sys.exit(0)


# Register the signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Verify Unity ML-Agents environment.')
    parser.add_argument('--env_path', type=str, required=True,
                        help='Path to the Unity environment executable')
    parser.add_argument('--no-graphics', action='store_true',
                        help='Run the environment in no-graphics mode')
    args = parser.parse_args()

    try:
        # Initialize Unity environment with the provided path using a retry mechanism
        print(f"Attempting to connect to environment at: {args.env_path}")

        # Create the environment with an improved retry mechanism for different worker_ids
        max_retries = 10  # Increased from 5 to 10
        env = None

        # Try a wider range of worker_ids to avoid port conflicts
        for worker_id in range(10, 30):  # Use higher worker_ids (10-29) to avoid common port conflicts
            try:
                print(f"Attempting to connect with worker_id={worker_id}...")
                # Use only supported parameters
                env = UnityEnvironment(
                    file_name=args.env_path, 
                    worker_id=worker_id, 
                    no_graphics=args.no_graphics
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

        # Reset the environment and print success message
        env_info = env.reset(train_mode=False)[brain_name]
        print("\nEnvironment reset successful")

        # Print environment information
        print("\nEnvironment Information:")
        print(f"Brain names: {env.brain_names}")
        print(f"Number of agents: {len(env_info.agents)}")
        print(f"State shape: {env_info.vector_observations.shape}")
        print(f"Action shape: ({brain.vector_action_space_size},)")

        print("\nEnvironment verification successful!")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        if 'env' in locals():
            try:
                env.close()
            except Exception as e:
                print(f"Error during environment cleanup: {e}")


if __name__ == "__main__":
    main()
