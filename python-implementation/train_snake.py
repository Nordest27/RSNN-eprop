import numpy as np
import scipy.sparse as sp
from snake_env import SnakeEnv
import time
import cv2
import matplotlib.pyplot as plt
from local_broadcast_srnn import LocalBroadcastSrnn

BOARD_SIZE = 5
VISIBLE_RANGE = 5
INPUT_SIZE = 3 * VISIBLE_RANGE**2


def build_agent():
    n_hidden = 256
    num_layers = 4
    num_neurons_list = [n_hidden for _ in range(num_layers)]
    output_size = 4
    input_connectivity = 0.5
    hidden_connectivity = 0.025
    output_connectivity = 0.3
    local_connectivity = 0.25

    srnn = LocalBroadcastSrnn(
        num_neurons_list=num_neurons_list,
        input_size=INPUT_SIZE,
        output_size=output_size,
        input_connectivity=input_connectivity,
        hidden_connectivity=hidden_connectivity,
        output_connectivity=output_connectivity,
        local_connectivity=local_connectivity,
        how_many_outputs=1,
        rl_gamma=0.99,
        default_learning_rate=1e-3,
        target_firing_rate=50,
        tau=20e-3,
        tau_out=10e-3,
        tau_adaptation=200e-3,
    )

    return srnn


def train_snake_agent(episodes=100000):
    env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE, wait_inc=7)
    agent = build_agent()

    best_reward = -np.inf
    best_run = []
    reward_history = []
    running_avg = []


    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)

    # --- 1. Training reward subplot ---
    line, = ax1.plot([], [], label='Total Reward', color='blue')
    avg_line, = ax1.plot([], [], label='Running Avg', color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()

    # --- 2. Value output subplot ---
    value_line, = ax2.plot([], [], color='green')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Value Output')
    ax2.set_title('Best Run Value Outputs')

    # --- 3. Action probabilities heatmap subplot ---
    prob_img = ax3.imshow(
        np.zeros((3, 1)),
        aspect='auto',
        cmap='viridis',
        origin='lower',
        vmin=0.0,
        vmax=1.0,
        extent=[0, 1, 0, 3]  # <<-- explicit extent (x from 0→1, y from 0→n_actions)
    )
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Action')
    ax3.set_title('Best Run Action Probabilities')
    fig.colorbar(prob_img, ax=ax3, fraction=0.02, pad=0.04)
    smoothing = 0.95
    avg = 0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        current_run = []
        current_values = []
        current_probs = []
        total_td = 0
        frame = 0
        
        input_vec = np.zeros((1, agent.total_neurons + agent.input_size), dtype=np.float32)
        input_vec[0, :INPUT_SIZE] = obs
        input_csr = sp.csr_matrix(input_vec)

        for _ in range(8):
            agent.input(input_csr)

        while not done:
            input_vec = np.zeros((1, agent.total_neurons + agent.input_size), dtype=np.float32)
            input_vec[0, :INPUT_SIZE] = obs
            input_csr = sp.csr_matrix(input_vec)

            agent.input(input_csr)

            action = agent.action()
            current_probs.append(agent.output_layers[0].policy_output.output())
            current_values.append(agent.output_layers[0].value_output.output()[0])

            possible_actions = np.sum(action, axis=0)
            most_voted_actions = [
                ai for ai in range(len(possible_actions))
                if possible_actions[ai] == max(possible_actions)
            ]
            action_label = np.random.choice(most_voted_actions)

            obs, reward, done = env.step(action_label)
            total_reward += reward

            td = agent.td_error_update(reward)
            # td = agent.receive_reward(reward)

            if env.wait_count == env.wait_inc:
                # agent.reset()
                for o in agent.output_layers:
                    o.policy_output.reset()
                    # o.reset()

            current_run.append(env.img(scale=50))

            total_td += td
            frame += 1
        
        agent.update_parameters()
        agent.reset()

        # --- Update if new best run ---
        if total_reward >= best_reward:
            best_reward = total_reward
            best_run = [img for img in current_run]

            # Update value plot
            value_line.set_data(range(len(current_values)), current_values)
            ax2.relim()
            ax2.autoscale_view()
            ax2.set_title(f'Best Run Value Outputs (Reward = {best_reward:.2f})')

            # --- Update heatmap plot with correct extent ---
            time_steps = len(current_probs)
            prob_img.set_data(np.array(current_probs).T)
            prob_img.set_extent([0, time_steps, 0, 4])  # <<-- fix horizontal scaling
            ax3.set_xlim(0, time_steps)
            ax3.set_ylim(0, 4)
            ax3.set_aspect('auto')
            ax3.set_title(f'Best Run Action Probabilities (Reward = {best_reward:.2f})')
            plt.pause(0.01)

        if (ep) % 50 == 0 and best_run:
            print(f"Replaying best run so far (reward = {best_reward:.2f})...")
            best_reward = -1
            for img in best_run:
                cv2.imshow("Snake", img)
                cv2.waitKey(1)
                time.sleep(0.01)

        print(f"Episode {ep+1} - Total reward: {total_reward:.2f} - Td Error: {total_td/frame:.3f} - Frame death: {frame}")

        reward_history.append(total_reward)
        if avg == 0:
            avg = total_reward
        else:
            avg = smoothing * avg + (1 - smoothing) * total_reward
        running_avg.append(avg)

        if ep % 10 == 0:
            reward_history = reward_history[-200:]
            running_avg = running_avg[-200:]
            axis = range(ep - len(reward_history), ep)
            line.set_data(axis, reward_history)
            avg_line.set_data(axis, running_avg)
            ax1.relim()
            ax1.autoscale_view()
            plt.pause(0.01)

    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    train_snake_agent()
