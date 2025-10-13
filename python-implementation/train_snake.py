import numpy as np
import scipy.sparse as sp
from snake_env import SnakeEnv
import time
import cv2
import matplotlib.pyplot as plt
from local_broadcast_srnn import LocalBroadcastSrnn

BOARD_SIZE = 3
VISIBLE_RANGE = 5
INPUT_SIZE = 3 * VISIBLE_RANGE**2 


def build_agent():
    n_hidden = 128
    num_layers = 4
    num_neurons_list = [n_hidden for _ in range(num_layers)]
    output_size = 4
    input_connectivity = 0.75
    hidden_connectivity = 0.015
    output_connectivity = 0.5
    local_connectivity = 0.3

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
        # tau_out=30e-3,
        target_firing_rate=40,
    )

    return srnn

def train_snake_agent(episodes=100000):
    env = SnakeEnv(size=BOARD_SIZE, visible_range=VISIBLE_RANGE)
    agent = build_agent()

    best_reward = -1
    best_run = []
    reward_history = []
    running_avg = []

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], label='Total Reward', color='blue')
    avg_line, = ax.plot([], [], label='Running Avg', color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Progress')
    ax.legend()

    smoothing = 0.95
    avg = 0

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        current_run = []
        total_td = 0
        frame = 0

        while not done:
            input_vec = np.zeros((1, agent.total_neurons + agent.input_size), dtype=np.float32)
            input_vec[0, :INPUT_SIZE] = obs
            input_csr = sp.csr_matrix(input_vec)
            
            for i in range(1):
                agent.input(input_csr)

            action = agent.action()
    
            possible_actions = np.sum(action, axis=0)
            most_voted_actions = [ai for ai in range(len(possible_actions)) if possible_actions[ai] == max(possible_actions)]
            #print(most_voted_actions)
            action_label = np.random.choice(most_voted_actions)

            obs, reward, done = env.step(action_label)
            # obs, reward, done = env.step(np.random.randint(0, 3))
            total_reward += reward - 1

            td = agent.td_error_update(reward)
            # td = agent.receive_reward(reward)
            agent.reset()
            for o in agent.output_layers:
                o.reset()

            current_run.append(env.img(scale=50))

            total_td += td
            frame += 1

        if total_reward >= best_reward:
            best_reward = total_reward
            best_run = [t for t in current_run]

        if (ep) % 250 == 0 and best_run:
            print(f"Replaying best run so far (reward = {best_reward:.2f})...")
            best_reward = -1
            for img in best_run:
                cv2.imshow("Snake", img)
                cv2.waitKey(1)
                time.sleep(0.1)

        print(f"Episode {ep+1} - Total reward: {total_reward:.2f} - Td Error: {total_td/frame:.3f} - Frame death: {frame}")

        reward_history.append(total_reward)
        if avg == 0:
            avg = total_reward
        else:
            avg = smoothing * avg + (1 - smoothing) * total_reward
        running_avg.append(avg)
        reward_history = reward_history[-200:]
        running_avg = running_avg[-200:]
        axis = range(ep-len(reward_history), ep)
        line.set_data(axis, reward_history)
        avg_line.set_data(axis, running_avg)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        agent.update_parameters()
        agent.reset()

    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    train_snake_agent()