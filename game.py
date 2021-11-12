import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
OLIVE = np.array([85, 127, 37])

BLOCK_SIZE = 20
SPEED = 30


class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(3*BLOCK_SIZE), self.head.y),
                      Point(self.head.x-(4*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = []
        self.turning_penalty = 0

        for _ in range(20):
            self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
            food = Point(x, y)
            if not food in self.snake:
                self.food.append(food)
                break

    def play_step(self, action, headless, slime_grid):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        reward = 0
        if not np.array_equal(action, [1, 0, 0]):
            if self.turning_penalty < 20:
                self.turning_penalty += 1
        else:
            self.turning_penalty = 0
        reward = -self.turning_penalty  # small penalty for making a turn

        # 3. check if game over
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -60
            return reward, game_over, self.score

        if self.frame_iteration > 30*len(self.snake):
            game_over = True
            return reward, game_over, self.score

        # 4. Penalise where snake has been recently
        # slime_trail_penalty = 5 * \
        #     slime_grid[int(self.head.x // BLOCK_SIZE),
        #                int(self.head.y // BLOCK_SIZE)]
        # reward -= slime_trail_penalty

        # 5. place new food or just move
        if self.head in self.food:
            self.food.remove(self.head)
            self.score += 1
            reward = 30
            self._place_food()
        else:
            self.snake.pop()

        # 6. update ui and clock
        if not headless:
            self._update_ui(slime_grid)
            self.clock.tick(SPEED)

        # 7. return game over and score
        return reward, game_over, self.score

    def is_snake_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        return self.is_snake_collision(pt)

    def _update_ui(self, slime_grid):
        self.display.fill(BLACK)

        for y in range(24):
            for x in range(32):
                slimeyness = slime_grid[x, y]
                pygame.draw.rect(self.display, OLIVE*slimeyness,
                                 pygame.Rect(x*20, y*20, BLOCK_SIZE, BLOCK_SIZE))

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        for food in self.food:
            pygame.draw.rect(self.display, RED, pygame.Rect(
                food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
