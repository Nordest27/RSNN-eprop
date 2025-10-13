import mss
import numpy as np
import cv2
import time
import scipy.sparse as sp
from local_broadcast_srnn import LocalBroadcastSrnn
from copy import deepcopy

def poisson_encode(image: np.ndarray, max_prob=1.0):
    """
    Convert an image to a binary spike train over time.
    image: pixel values [0, 1]
    returns: flattened spike train
    """
    flat_image = image.flatten()
    spike_probs = max_prob*flat_image / (max(flat_image))
    spikes = np.random.rand(len(flat_image)) < spike_probs
    return spikes


def build_agent():
    n_hidden = 128
    num_layers = 5
    num_neurons_list = [n_hidden for _ in range(num_layers)]
    input_size = 100*100*3
    output_size = 5
    input_connectivity = 0.01
    hidden_connectivity = 0.025
    output_connectivity = 0.25
    local_connectivity = 0.13
    # srnn = SimpleBroadcastSrnn(
    srnn = LocalBroadcastSrnn(
        num_neurons_list=num_neurons_list, 
        input_size=input_size, 
        output_size=output_size,
        input_connectivity=input_connectivity,
        hidden_connectivity=hidden_connectivity,
        output_connectivity=output_connectivity,
        local_connectivity=local_connectivity,
        rl_gamma=0.9,
        default_learning_rate=1e-3,
        # self_predict=True,
        # target_firing_rate=13,
    )
    return srnn


class ScreenCaptureMonitor:
    def __init__(self, region, step=20):
        """
        region: dict with 'top', 'left', 'width', 'height'
        step: how many pixels to move per adjustment
        """
        self.sct = mss.mss()
        self.region = region
        self.initial_region = deepcopy(region)
        self.step = step

        # Get screen dimensions (first monitor)
        monitor = self.sct.monitors[1]  # 0 = all monitors, 1 = primary
        self.screen_width = monitor["width"]
        self.screen_height = monitor["height"]
        self.srnn = build_agent()

    def capture_frame(self):
        """Capture a single frame and return as a BGR numpy array"""
        screenshot = self.sct.grab(self.region)
        frame = np.array(screenshot)  # BGRA format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR
        return frame

    def clamp_region(self):
        """Ensure region stays inside the screen"""
        if self.region["left"] < 0:
            self.region["left"] = 0
        if self.region["top"] < 0:
            self.region["top"] = 0
        if self.region["left"] + self.region["width"] > self.screen_width:
            self.region["left"] = self.screen_width - self.region["width"]
        if self.region["top"] + self.region["height"] > self.screen_height:
            self.region["top"] = self.screen_height - self.region["height"]

    def move(self, direction):
        """Shift the capture region and clamp to screen"""
        if direction == "up":
            self.region["top"] -= self.step
        elif direction == "down":
            self.region["top"] += self.step
        elif direction == "left":
            self.region["left"] -= self.step
        elif direction == "right":
            self.region["left"] += self.step

        self.clamp_region()

    def live_monitor(self, fps=10):
        """Continuously capture and show the region"""
        frame_time = 1.0 / fps
        frame = self.capture_frame()
        frame_count = 0
        try:
            while True:
                frame_count += 1
                start = time.time()
                
                input_size = self.srnn.input_size
                total_size = self.srnn.input_size + self.srnn.total_neurons
                input_row = np.zeros((1, total_size))
                input_row[0, :input_size] = poisson_encode(frame)
                input_csr = sp.csr_matrix(input_row)

                self.srnn.input(input_csr)

                action = self.srnn.action()
                action_label = np.argmax(action)

                if action_label < 4:
                    self.move(["up", "down", "left", "right"][action_label])
                
                frame = self.capture_frame()

                reward = frame.mean()/255
                adv = self.srnn.td_error_update(reward)
                # Example: numerical data (mean intensity)
                print(f"Mean pixel intensity: {reward:.2f}, Adv: {adv:.2f}, Value: {self.srnn.ac_output_layer.value_output.output()[0]:.2f}")

                if frame_count % 10 == 0:
                    self.srnn.update_parameters()
                    # print(self.region)
                    # self.region = deepcopy(self.initial_region)
                    # print(self.region)
                    # self.srnn.reset()
                    # print("__________________________")
                
                # --- Show the frame in a window ---
                cv2.imshow("Screen Region Monitor", frame)

                # --- Controls ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):  # Quit
                    break
                elif key == ord("w"):  # Move up
                    self.move("up")
                elif key == ord("s"):  # Move down
                    self.move("down")
                elif key == ord("a"):  # Move left
                    self.move("left")
                elif key == ord("d"):  # Move right
                    self.move("right")


                # Maintain FPS
                elapsed = time.time() - start
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)


        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # Initial screen region (adjust values as needed)
    region = {
        "top": np.random.randint(0, 500),
        "left": np.random.randint(0, 500),
        "width": 100,
        "height": 100
    }

    capture = ScreenCaptureMonitor(region, step=20)

    print("Starting live monitor... Press 'q' to quit, WASD to move region")
    capture.live_monitor(fps=60)
