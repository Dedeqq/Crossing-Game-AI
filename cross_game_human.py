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
SPEED = 2

class CrossingGame:
    
    def __init__(self):
        # window size
        self.width = 860
        self.height = 480
        
        # display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
       
        # player block
        self.player = Square(0, self.height/2)
        
        # directions
        self.directions={"right":0, "down":1, "left":2, "up": 3 }
        self.player_direction = self.directions["right"]
        self.wall_direction = self.directions["down"]
        
        # moving walls
        self.walls=[self.place_wall(x) for x in range(60,860,80)]
        self.score = 0
    
    def place_wall(self, x):
        # create walls
        wall=[]
        length=0
        empty=random.randint(0,1)
        while length<self.height-BLOCK_SIZE:
            empty=(empty+1)%2
            size=random.randint(3,4)
            if empty==0:
                for i in range(size):
                    if length==self.height-2*BLOCK_SIZE: return wall
                    wall.append(Square(x, length))
                    length+=BLOCK_SIZE
            else: length+=size*BLOCK_SIZE
        return wall
            
    def move_wall(self, wall):
        # move wall by one block
        self.wall_direction = (self.wall_direction+2)%4
        helper=[]
        if self.wall_direction==self.directions["down"]:
            for square in wall: helper.append(Square(square.x, (square.y+BLOCK_SIZE)%self.height))
        if self.wall_direction==self.directions["up"]:
            for square in wall: helper.append(Square(square.x, (square.y-BLOCK_SIZE)%self.height))
        return helper
                    
        
    def next_frame(self):
        # collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player_direction = self.directions["left"]
                elif event.key == pygame.K_RIGHT:
                    self.player_direction = self.directions["right"]
                elif event.key == pygame.K_UP:
                    self.player_direction = self.directions["up"]
                elif event.key == pygame.K_DOWN:
                    self.player_direction = self.directions["down"]
        
        # move walls and player
        self.player_move(self.player_direction)
        self.walls=[self.move_wall(wall) for wall in self.walls]
        
        # check for collisions
        game_over = False
        if self.game_over_check():
            game_over = True
            return game_over, self.score
        
        # update score
        if self.score<self.player.x/BLOCK_SIZE: self.score=self.player.x//BLOCK_SIZE 
        
        # update display and clock
        self.update_display()
        self.clock.tick(SPEED)
        
        # return game over and score
        return game_over, self.score
    
    def game_over_check(self):
        for wall in self.walls:
            if self.player in wall: return True
        if self.player.x<0 or self.player.x>860: return True
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
        
    def player_move(self, direction):
        x = self.player.x
        y = self.player.y
        if direction == self.directions["right"]:
            x += BLOCK_SIZE
        elif direction == self.directions["left"]:
            x -= BLOCK_SIZE
        elif direction == self.directions["down"]:
            y += BLOCK_SIZE
        elif direction ==self.directions["up"]:
            y -= BLOCK_SIZE
            
        self.player = Square(x, y%self.height)
            

if __name__ == '__main__':
    game = CrossingGame()
    
    # game loop
    while True:
        game_over, score = game.next_frame()
        
        if game_over == True:
            break
        
    print('Game over, your score: ', score)
        
        
    pygame.quit()