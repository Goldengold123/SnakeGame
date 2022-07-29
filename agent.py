import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import LinearQNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 100
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_L = Point(head.x - BLOCK_SIZE, head.y)
        point_R = Point(head.x + BLOCK_SIZE, head.y)
        point_U = Point(head.x, head.y - BLOCK_SIZE)
        point_D = Point(head.x, head.y + BLOCK_SIZE)

        dir_L = (game.direction == Direction.LEFT)
        dir_R = (game.direction == Direction.RIGHT)
        dir_U = (game.direction == Direction.UP)
        dir_D = (game.direction == Direction.DOWN)

        # state =
        # [danger straight, danger right, danger left,
        # direction left, direction right, direction up , direction down,
        # food left, food right, food up, food down]
        state = [
            # danger straight
            (
                    (dir_L and game.is_collision(point_L)) or
                    (dir_R and game.is_collision(point_R)) or
                    (dir_U and game.is_collision(point_U)) or
                    (dir_D and game.is_collision(point_D))
            ),

            # danger right
            (
                    (dir_L and game.is_collision(point_U)) or
                    (dir_R and game.is_collision(point_D)) or
                    (dir_U and game.is_collision(point_R)) or
                    (dir_D and game.is_collision(point_L))
            ),

            # danger left
            (
                    (dir_L and game.is_collision(point_D)) or
                    (dir_R and game.is_collision(point_U)) or
                    (dir_U and game.is_collision(point_L)) or
                    (dir_D and game.is_collision(point_R))
            ),

            # move direction
            dir_L, dir_R, dir_U, dir_D,

            # food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
        final_move[move] = 1
        return final_move


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory / experience replay / replay memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game:", agent.n_games, "Score:", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)


if __name__ == '__main__':
    train()
