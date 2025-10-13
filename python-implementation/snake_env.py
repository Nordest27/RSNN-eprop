
from re import S
import numpy as np
import random
import time
import cv2

class SnakeEnv:
    def __init__(self, size=28, visible_range=7):
        assert visible_range % 2 == 1, "visible_range must be odd"
        self.size = size
        self.visible_range = visible_range
        self.reset()

    def reset(self):
        self.snake = [(self.size // 2, self.size // 2)]

        self.dir_idx = 1
        self.direction = 'up'

        self.spawn_apple()
        self.done = False
        self.steps_since_last_apple = 0
        return self.get_observation()

    def spawn_apple(self):
        empty_cells = [(i, j) for i in range(self.size)
                       for j in range(self.size) if (i, j) not in self.snake]
        if len(empty_cells) > 0:
            self.apple = random.choice(empty_cells)
        else:
            self.done = True
        #self.apple = empty_cells[15%len(empty_cells)]

    def step(self, action):
        if self.done:
            raise Exception("Environment needs reset. Call env.reset().")

        dirs = ['left', 'up', 'right', 'down']
        new_dir = dirs[action]

        # if (self.direction == 'up' and new_dir == 'down') or \
        #    (self.direction == 'down' and new_dir == 'up') or \
        #    (self.direction == 'left' and new_dir == 'right') or \
        #    (self.direction == 'right' and new_dir == 'left'):
        #     new_dir = self.direction

        self.direction = new_dir

        head_y, head_x = self.snake[0]
        if self.direction == 'up':
            head_y -= 1
        elif self.direction == 'down':
            head_y += 1
        elif self.direction == 'left':
            head_x -= 1
        elif self.direction == 'right':
            head_x += 1

        # if head_y < 0:
        #     head_y = self.size-1
        # elif head_y >= self.size:
        #     head_y = 0
        # elif head_x < 0:
        #     head_x = self.size-1
        # elif head_x >= self.size:
        #     head_x = 0
            
        new_head = (head_y, head_x)

        if (head_y < 0 or head_y >= self.size or
            head_x < 0 or head_x >= self.size or
            new_head in self.snake
        ):
            self.done = True
            return self.get_observation(), -100, True

        self.snake.insert(0, new_head)

        reward = 0.0
        if new_head == self.apple:
            reward = 100.0
            self.spawn_apple()
            self.steps_since_last_apple = 0
        else:
            self.snake.pop()
            self.steps_since_last_apple += 1
        
        if self.steps_since_last_apple > self.size**2:
            self.done = True
            return self.get_observation(), 0.0, True
    
        # reward = 0
        # if self.direction == "right":
        #     reward = 1

        return self.get_observation(), reward, self.done

    def get_observation(self):
        """
        Returns a (4, visible_range, visible_range) tensor:
        0: snake body
        1: head
        2: apple
        3: walls
        """
        v = self.visible_range
        r = v // 2
        head_y, head_x = self.snake[0]

        obs = np.zeros((3, v, v), dtype=np.float32)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx
                local_y = dy + r
                local_x = dx + r

                # check walls
                if y < 0 or y >= self.size or x < 0 or x >= self.size:
                    obs[2, local_y, local_x] = 1.0
                    continue

                # body
                if (y, x) in self.snake[1:]:
                    obs[0, local_y, local_x] = 1.0

                # # head
                # if (y, x) == (head_y, head_x):
                #     obs[1, local_y, local_x] = 1.0

                # apple
                if (y, x) == self.apple:
                    obs[1, local_y, local_x] = 1.0

        return obs.flatten()

    def img(self, scale=10):
        img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for y, x in self.snake[1:]:
            img[y, x] = [0, 255, 0]
        head_y, head_x = self.snake[0]
        img[head_y, head_x] = [0, 155, 0]
        ay, ax = self.apple
        img[ay, ax] = [0, 0, 255]
        return cv2.resize(img, (self.size * scale, self.size * scale), interpolation=cv2.INTER_NEAREST)

    def local_img(self, scale=10):
        """
        Returns a local RGB image centered on the snake's head,
        matching the visible_range used in get_observation().
        Channels:
            Green  = snake body
            Dark green = snake head
            Red    = apple
            Gray   = walls
        """
        v = self.visible_range
        r = v // 2
        head_y, head_x = self.snake[0]

        img = np.zeros((v, v, 3), dtype=np.uint8)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                y = head_y + dy
                x = head_x + dx
                local_y = dy + r
                local_x = dx + r

                # walls
                if y < 0 or y >= self.size or x < 0 or x >= self.size:
                    img[local_y, local_x] = [100, 100, 100]  # gray
                    continue

                # body
                if (y, x) in self.snake[1:]:
                    img[local_y, local_x] = [0, 255, 0]

                # head
                if (y, x) == (head_y, head_x):
                    img[local_y, local_x] = [0, 155, 0]

                # apple
                if (y, x) == self.apple:
                    img[local_y, local_x] = [0, 0, 255]

        # upscale for visualization
        return cv2.resize(img, (v * scale, v * scale), interpolation=cv2.INTER_NEAREST)

    def render_cv2(self, scale=10):
        img = self.img()
        cv2.imshow("Snake", img)
        cv2.waitKey(1)
