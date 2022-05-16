import numpy as np
import pygame
import random
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)
    
Square = namedtuple('Square', 'x, y')

# colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN = (0,200,0)
BLACK = (0,0,0)


BLOCK_SIZE = 20
SPEED = 20

class CrossingGame:
    
    def __init__(self):
        # window size
        self.width = 860
        self.height = 480
        
        # games won in a row
        self.won_games=0
        
        # display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset_game()
    
    def reset_game(self):
        # player block
        self.player = Square(0, self.height/2)
        
        # score
        self.score=0
        
        # directions
        self.directions={"right":0, "down":1, "left":2, "up": 3 }
        self.player_direction = self.directions["right"]
        
        # moving walls
        self.walls=[self.place_wall(x) for x in range(60,860,80)]
        
        # no infinite games
        self.number_of_moves=0
    
    def place_wall(self, x):
        # create walls
        wall=[]
        length=0
        empty=random.randint(0,1)
        while length<self.height-BLOCK_SIZE:
            empty=(empty+1)%2
            size=random.randint(2,3)
            if empty==0:
                for i in range(size):
                    if length==self.height-2*BLOCK_SIZE: return wall
                    wall.append(Square(x, length))
                    length+=BLOCK_SIZE
            else: length+=size*BLOCK_SIZE
        return wall
            
    def move_wall(self, wall):
        # move wall by one block
        helper=[]
        for square in wall: helper.append(Square(square.x, (square.y+BLOCK_SIZE)%self.height))
        return helper
                    
        
    def next_frame(self, action):
        self.number_of_moves+=1
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move walls and player
        self.player_move(action)
        self.walls=[self.move_wall(wall) for wall in self.walls]
        
        # check for collisions
        reward=0
        game_over = False
        if self.game_over_check() or self.number_of_moves>200:
            game_over = True
            reward=-10
            return reward, game_over, self.score
        
        if self.player.x==860: 
            reward=20
            game_over = True
            return reward, game_over, self.score
        
        # update score
        if self.score<self.player.x/BLOCK_SIZE: 
            self.score=self.player.x//BLOCK_SIZE
            reward=10
        
        # update display and clock
        self.update_display()
        self.clock.tick(SPEED)
        # return game over and score
        return reward, game_over, self.score

    
    def game_over_check(self, block=None):
        if block is None: block=self.player
        for wall in self.walls:
            if block in wall: return True
        if block.x<0: return True
        return False
        
    def update_display(self):
        self.display.fill(BLACK)
        
        pygame.draw.rect(self.display, RED, pygame.Rect(self.player.x, self.player.y, BLOCK_SIZE, BLOCK_SIZE))
        
        for wall in self.walls: 
            for pt in wall:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def player_move(self, action):
        # action is binary array of length four with exactly one 1
        # position of 1 defines the direction [right, down, left, up]
        
        x = self.player.x
        y = self.player.y
        
        if np.array_equal(action, [1,0,0,0]):
            self.player_direction == self.directions["right"]
            x += 20
        elif np.array_equal(action, [0,1,0,0]):
            self.player_direction == self.directions["down"]
            y += 20
        elif np.array_equal(action, [0,0,1,0]):
            self.player_direction == self.directions["left"]
            x -= 20
        elif np.array_equal(action, [0,0,0,1]):
            self.player_direction == self.directions["up"]
            y -= 20
            
        self.player = Square(x, y%self.height)