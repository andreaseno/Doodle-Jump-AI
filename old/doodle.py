import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'

import pygame 
import sys
import random
import math
import time
import multiprocessing
import numpy as np
from icecream import ic


"""Global variables"""

RENDER = True
# default resolution: 360 x 640
RESOLUTION = WIDTH, HEIGHT = 360, 640
TITLE = "Doodle Jump"
TIME_SPEED = 1
NUM_PLAYERS = 8

gravity = 0.15
file_name = "high_score"
background_color = (250, 248, 239)
x_scale = WIDTH / 360
y_scale = HEIGHT / 640
gravity *= y_scale

""" Objects"""

class Doodle():
    pygame.init()
    if RENDER:
        pygame.font.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption(TITLE)
    
    class Player():
        jump_force = 8 * y_scale
        max_x_speed = 16 * x_scale
        x_acceleration = 0.15 * x_scale
        color = (255, 255, 0)
        color2 = (0, 255, 255)
        height = 32 * y_scale
        width = 32 * y_scale

        def __init__(self):
            self.y = HEIGHT - self.height
            self.x = (WIDTH - self.width) // 2
            self.y_speed = -self.jump_force
            self.x_speed = 0
            self.direction = 0
            self.moving_direction = 0
            self.score = 0

        def move(self, left_key_pressed, right_key_pressed, time_scale):
            # simulate gravity
            self.y_speed += gravity * time_scale
            self.y += self.y_speed * time_scale

            # change player's speed
            if left_key_pressed == right_key_pressed:
                self.x_speed = (max(0, math.fabs(self.x_speed) - (
                            self.x_acceleration / 2) * time_scale)) * self.moving_direction
                self.direction = 0
            elif left_key_pressed:
                self.x_speed = max(-self.max_x_speed, self.x_speed - self.x_acceleration * time_scale)
                self.direction = -1
                self.moving_direction = -1
            elif right_key_pressed:
                self.x_speed = min(self.max_x_speed, self.x_speed + self.x_acceleration * time_scale)
                self.direction = 1
                self.moving_direction = 1

            # move player
            self.x += self.x_speed * time_scale
            if self.x + self.width + 20 < 0:
                self.x = WIDTH
            if self.x > WIDTH:
                self.x = -20 - self.width

        def jump(self):
            self.y_speed = -self.jump_force

        def high_jump(self):
            self.y_speed = -self.jump_force * 2

        def draw(self):
            pygame.draw.rect(Doodle.screen, self.color, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(Doodle.screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 1)
            if self.direction <= 0:
                pygame.draw.rect(Doodle.screen, self.color2,
                                (self.x + 6 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale))
                pygame.draw.rect(Doodle.screen, (0, 0, 0), (self.x + 6 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale),
                                1)
            if self.direction >= 0:
                pygame.draw.rect(Doodle.screen, self.color2,
                                (self.x + self.width - 13 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale))
                pygame.draw.rect(Doodle.screen, (0, 0, 0),
                                (self.x + self.width - 13 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale), 1)
            if self.direction == 1:
                pygame.draw.rect(Doodle.screen, self.color2,
                                (self.x + self.width - 15 * y_scale, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale))
                pygame.draw.rect(Doodle.screen, (0, 0, 0),
                                (self.x + self.width - 15 * y_scale, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale), 1)
            elif self.direction == -1:
                pygame.draw.rect(Doodle.screen, self.color2, (self.x, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale))
                pygame.draw.rect(Doodle.screen, (0, 0, 0), (self.x, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale), 1)
            else:
                pygame.draw.rect(Doodle.screen, self.color2,
                                (self.x + 4 * y_scale, self.y + 18 * y_scale, 24 * y_scale, 7 * y_scale))
                pygame.draw.rect(Doodle.screen, (0, 0, 0),
                                (self.x + 4 * y_scale, self.y + 18 * y_scale, 24 * y_scale, 7 * y_scale), 1)


    class Platform():
        width = 64 * y_scale
        height = 16 * y_scale

        def __init__(self, y, score):
            self.x = random.randint(0, int(WIDTH - self.width))
            self.y = y
            # platform types
            if score < 500:
                self.type = 0
            elif score < 1500:
                self.type = random.choice([0, 0, 0, 0, 0, 0, 1, 1])
            elif score < 2500:
                self.type = random.choice([0, 0, 0, 0, 1, 1, 1, 1])
            elif score < 3500:
                self.type = random.choice([0, 0, 0, 1, 1, 1, 1, 2])
            elif score < 5000:
                self.type = random.choice([0, 0, 1, 1, 1, 2, 3])
            else:
                self.type = random.choice([1, 1, 1, 1, 1, 2, 3, 3])

            # decide if platform has spring on top
            # decide initial direction the platform if it moves
            if self.type == 0:
                self.color = (63, 255, 63)
                self.direction = 0
                self.alpha = -1
                self.have_spring = random.choice([False] * 15 + [True])
            elif self.type == 1:
                self.color = (127, 191, 255)
                self.direction = random.choice([-1, 1]) * y_scale
                self.have_spring = random.choice([False] * 15 + [True])
                self.alpha = -1
            elif self.type == 2:
                self.color = (191, 0, 0)
                self.direction = 0
                self.have_spring = False
                self.alpha = 0
            else:
                self.color = background_color
                self.direction = 0
                self.have_spring = False
                self.alpha = -1

        def move(self, time_scale):
            self.x += self.direction * time_scale
            if self.x < 0:
                self.direction *= -1
                self.x = 0
            if self.x + self.width > WIDTH:
                self.direction *= -1
                self.x = WIDTH - self.width

        def draw(self, time_scale):
            pygame.draw.rect(Doodle.screen, self.color, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(Doodle.screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 1)
            if self.alpha > 0:
                self.alpha += 16 * time_scale
                s = pygame.Surface((self.width, self.height))
                s.set_alpha(self.alpha)
                s.fill(background_color)
                Doodle.screen.blit(s, (self.x, self.y))


    class Spring():
        width = 16 * y_scale
        height = 8 * y_scale
        color = (127, 127, 127)

        def __init__(self, platform):
            self.x = platform.x + platform.width // 2 - self.width // 2 + random.randint(
                -platform.width // 2 + self.width // 2, platform.width // 2 - self.width // 2)
            self.y = platform.y - self.height
            self.direction = platform.direction
            self.left_edge = self.x - platform.x
            self.right_edge = WIDTH - ((platform.x + platform.width) - (self.x + self.width))

        def move(self, time_scale):
            self.x += self.direction * time_scale
            if self.x < self.left_edge:
                self.direction *= -1
                self.x = self.left_edge
            if self.x + self.width > self.right_edge:
                self.direction *= -1
                self.x = self.right_edge - self.width

        def draw(self):
            pygame.draw.rect(Doodle.screen, self.color, (self.x, self.y, self.width, self.height))
            pygame.draw.rect(Doodle.screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 1)

    def __init__(self):
        self.read_high_score()
        self.new_game() # create new game and play

    def play(self):
        while True:
            self.simulate(self.player, self.platforms, self.springs, self.time_scale)
            if self.player.score > self.high_score:
                self.high_score = self.player.score

            self.new_platforms(self.player)

            # STOP
            if self.is_game_over():
                # pygame.quit()
                break

            # Prevent the code from running too fast during a simulation
            if not RENDER: time.sleep(0.01)

            self.time_scale = (pygame.time.get_ticks() - self.prev_time) / 10 * TIME_SPEED
            self.prev_time = pygame.time.get_ticks()
            
    def simulate(self, player, platforms, springs, time_scale):
        (left_key_pressed, right_key_pressed) = self.get_event()
        player.move(left_key_pressed, right_key_pressed, time_scale)

        # check if player go above half of screen's height
        if player.y < HEIGHT // 2 - player.height:
            movement = HEIGHT // 2 - player.height - player.y
            player.y = HEIGHT // 2 - player.height
        else:
            movement = 0
        player.score += movement / 4 / y_scale
        self.update_game(player, platforms, springs, time_scale, movement)

        if RENDER:
            self.render_game(Doodle.screen, player, platforms, springs, time_scale)
            
    def update_game(self, player, platforms, springs, time_scale, movement):
        i = 0
        while i < len(platforms):
            platforms[i].y += movement
            platforms[i].move(time_scale)
            # check if player fall on a platform
            if player.y_speed >= 0 and player.x < platforms[i].x + self.Platform.width and player.x + self.Player.width > platforms[
                i].x and player.y + self.Player.height <= platforms[
                i].y + time_scale * player.y_speed and player.y + self.Player.height >= platforms[i].y:
                if platforms[i].type != 2:
                    player.y = platforms[i].y - self.Player.height
                    player.jump()
                    if platforms[i].type == 3:
                        del platforms[i]
                        i -= 1
                else:
                    platforms[i].alpha = max(1, platforms[i].alpha)
                    if platforms[i].alpha >= 255:
                        del platforms[i]
                        i -= 1
            i += 1

        for spring in springs:
            spring.move(time_scale)
            spring.y += movement
            # check if player fall on a spring
            if player.y_speed >= 0 and player.x < spring.x + self.Spring.width and player.x + self.Player.width > spring.x and player.y + self.Player.height >= spring.y and player.y <= spring.y + self.Spring.height:
                player.high_jump()
                
    def render_game(self, screen, player, platforms, springs, time_scale):
        screen.fill(background_color)

        for platform in platforms:
            platform.draw(time_scale)

        for spring in springs:
            spring.draw()

        player.draw()

        #print scores
        font = pygame.font.SysFont("Comic Sans MS", int(24 * y_scale))
        text = font.render("Score:", True, (0, 0, 0))
        text2 = font.render(str(int(player.score)), True, (0, 0, 0))
        text3 = font.render("Best:", True, (0, 0, 0))
        text4 = font.render(str(int(self.high_score)), True, (0, 0, 0))
        text_width = max(text3.get_width(), text4.get_width())
        screen.blit(text, (10 * y_scale, 0))
        screen.blit(text2, (10 * y_scale, 24 * y_scale))
        screen.blit(text3, (WIDTH - text_width - 10 * y_scale, 0))
        screen.blit(text4, (WIDTH - text_width - 10 * y_scale, 24 * y_scale))
        pygame.display.update()
            
    # set up new game
    def new_game(self):
        self.player = self.Player()
        self.platforms = [self.Platform(HEIGHT - 1, 0)]
        self.platforms[0].x = 0
        self.platforms[0].width = WIDTH
        self.platforms[0].color = (0, 0, 0)
        self.springs = []
        self.time_scale = 1
        self.prev_time = pygame.time.get_ticks()
        self.play()
        
    def is_game_over(self):
        if self.player.score == 0 and self.player.y + self.Player.height > HEIGHT - 2:
            self.player.y = HEIGHT - 2 - self.Player.height
            self.player.jump()
            return False
        elif self.player.y > HEIGHT:
            return True
        return False
    
    def game_over(self):
        print('stop')
        pygame.quit()
        
    
    def read_high_score(self):
        # if file doesn't exist
        if os.path.isfile(file_name) == False:
            self.high_score = 0
            with open(file_name, "w") as f:
                f.write("0")
        else:
            with open(file_name, "r+") as f:    
                self.high_score = int(f.read())
                
    def get_event(self):
        if RENDER:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    file = open(file_name, "w")
                    file.write(str(int(self.high_score)))
                    file.close()
                    pygame.quit()
                    sys.exit()
        pressed = pygame.key.get_pressed()
        left_key_pressed = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
        right_key_pressed = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
        return left_key_pressed, right_key_pressed
    
    def new_platforms(self, player):
        # as the score increaces, the gap between platforms increaces
        if player.score < 500:
            gap_lower_bound, gap_upper_bound = 24, 48
        elif player.score < 1500:
            gap_lower_bound, gap_upper_bound = 26, 52
        elif player.score < 2500:
            gap_lower_bound, gap_upper_bound = 28, 56
        elif player.score < 3500:
            gap_lower_bound, gap_upper_bound = 30, 60
        elif player.score < 5000:
            gap_lower_bound, gap_upper_bound = 32, 64
        else:
            gap_lower_bound, gap_upper_bound = 34, 68

        # delete platforms below the screen
        i = 0
        while i < len(self.platforms):
            if self.platforms[i].y > HEIGHT:
                del self.platforms[i]
            i += 1
        i = 0

        # delete springs below the screen
        while i < len(self.springs):
            if self.springs[i].y > HEIGHT:
                del self.springs[i]
            i += 1

        # generate platforms & springs
        while self.platforms[-1].y + self.platforms[-1].height >= 0:
            gap = random.randint(gap_lower_bound, gap_upper_bound) * y_scale
            platform = self.Platform(self.platforms[-1].y - gap, player.score)

            # can't have 3 fake platforms in a row
            if not (platform.type == 2 and self.platforms[-1].type == 2 and self.platforms[-2].type == 2):
                self.platforms.append(platform)
            # draw a spring if the platform have it
            if platform.have_spring:
                self.springs.append(self.Spring(platform))
                
    
def worker_fn(moves: list[int] = None):
    print('Start')
    doodle = Doodle()
    print('Done')
    return doodle.high_score
    
if __name__ == '__main__':
    np.random.seed(1024)
    random.seed(2048)
    
    random_moves = [list(np.random.randint(-1,1,100)) for _ in range(NUM_PLAYERS)]
    ic(len(random_moves))
    
    with multiprocessing.Pool() as pool:
        # get high score for each random sequence
        results = pool.map(worker_fn, random_moves)
        
    print(results)