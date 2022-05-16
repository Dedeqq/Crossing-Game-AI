import torch
import random
import numpy as np
from collections import deque
from corss_game_ai import CrossingGame, Square
from nn_model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(14, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.player
        Square_l1 = Square(head.x - 20, head.y-80)
        Square_l2 = Square(head.x - 20, head.y-60)
        Square_l3 = Square(head.x - 20, head.y-40)
        Square_l4 = Square(head.x - 20, head.y-20)
        Square_r1 = Square(head.x + 20, head.y-80)
        Square_r2 = Square(head.x + 20, head.y-60)
        Square_r3 = Square(head.x + 20, head.y-40)
        Square_r4 = Square(head.x + 20, head.y-20)
        Square_u = Square(head.x, head.y-20)
        
        dir_l = game.player_direction == game.directions["left"]
        dir_r = game.player_direction== game.directions["right"]
        dir_u = game.player_direction== game.directions["up"]
        dir_d = game.player_direction== game.directions["down"]

        state = [
            # Danger left 1
            game.game_over_check(Square_l1),
            # Danger left 2
            game.game_over_check(Square_l2),
            # Danger left 3
            game.game_over_check(Square_l3),
            # Danger left 4
            game.game_over_check(Square_l4),
            
            # Danger right 1
            game.game_over_check(Square_r1),
            # Danger right 2
            game.game_over_check(Square_r2),
            # Danger right 3
            game.game_over_check(Square_r3),
            # Danger right 4
            game.game_over_check(Square_r4),
            
            # Danger up
            game.game_over_check(Square_u),            
            
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Distance from finish lane
            (860-game.player.x)/20]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        finalmove = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            finalmove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalmove[move] = 1

        return finalmove


def train():
    total_score = 0
    record = 0
    agent = Agent()
    game = CrossingGame()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        finalmove = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.next_frame(finalmove)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, finalmove, reward, state_new, done)

        # remember
        agent.remember(state_old, finalmove, reward, state_new, done)

        if done:
            # train long memory
            game.reset_game()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record: record = score

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

if __name__ == '__main__':
    train()