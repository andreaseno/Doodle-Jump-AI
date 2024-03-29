import pygame, sys, random, math, os, time, gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from icecream import ic


debug = False
RENDER = True
# default resolution: 360 x 640
RESOLUTION = WIDTH, HEIGHT = 360, 640
TITLE = "Doodle Jump"
TIME_SPEED = 1

pygame.init()
gravity = 0.15
file_name = "high_score"
background_color = (250, 248, 239)
x_scale = WIDTH / 360
y_scale = HEIGHT / 640
if RENDER:
    pygame.font.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption(TITLE)

gravity *= y_scale


class Player():
    jump_force = 8 * y_scale
    max_x_speed = 8 * x_scale
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

    def move(self, action, time_scale):
        # simulate gravity
        self.y_speed += gravity * time_scale
        self.y += self.y_speed * time_scale

        # change player's speed
        if action == 1:
            self.x_speed = (max(0, math.fabs(self.x_speed) - (
                        self.x_acceleration / 2) * time_scale)) * self.moving_direction
            self.direction = 0
        elif action == 0:
            self.x_speed = max(-self.max_x_speed, self.x_speed - self.x_acceleration * time_scale)
            self.direction = -1
            self.moving_direction = -1
        elif action == 2:
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
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 1)
        if self.direction <= 0:
            pygame.draw.rect(screen, self.color2,
                             (self.x + 6 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x + 6 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale),
                             1)
        if self.direction >= 0:
            pygame.draw.rect(screen, self.color2,
                             (self.x + self.width - 13 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0),
                             (self.x + self.width - 13 * y_scale, self.y + 6 * y_scale, 7 * y_scale, 7 * y_scale), 1)
        if self.direction == 1:
            pygame.draw.rect(screen, self.color2,
                             (self.x + self.width - 15 * y_scale, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0),
                             (self.x + self.width - 15 * y_scale, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale), 1)
        elif self.direction == -1:
            pygame.draw.rect(screen, self.color2, (self.x, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y + 18 * y_scale, 15 * y_scale, 7 * y_scale), 1)
        else:
            pygame.draw.rect(screen, self.color2,
                             (self.x + 4 * y_scale, self.y + 18 * y_scale, 24 * y_scale, 7 * y_scale))
            pygame.draw.rect(screen, (0, 0, 0),
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
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 1)
        if self.alpha > 0:
            self.alpha += 16 * time_scale
            s = pygame.Surface((self.width, self.height))
            s.set_alpha(self.alpha)
            s.fill(background_color)
            screen.blit(s, (self.x, self.y))


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
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 1)




def game_over():
    pass


def get_event(high_score):
    if RENDER:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                file = open(file_name, "w")
                file.write(str(int(high_score)))
                file.close()
                pygame.quit()
                sys.exit()
    


def update_game(player, platforms, springs, time_scale, movement):
    i = 0
    while i < len(platforms):
        platforms[i].y += movement
        platforms[i].move(time_scale)
        # check if player fall on a platform
        if player.y_speed >= 0 and player.x < platforms[i].x + Platform.width and player.x + Player.width > platforms[
            i].x and player.y + Player.height <= platforms[
            i].y + time_scale * player.y_speed and player.y + Player.height >= platforms[i].y:
            if platforms[i].type != 2:
                player.y = platforms[i].y - Player.height
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
        if player.y_speed >= 0 and player.x < spring.x + Spring.width and player.x + Player.width > spring.x and player.y + Player.height >= spring.y and player.y <= spring.y + Spring.height:
            player.high_jump()


def render_game(screen, player, platforms, springs, time_scale, high_score):
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
    text4 = font.render(str(int(high_score)), True, (0, 0, 0))
    text_width = max(text3.get_width(), text4.get_width())
    screen.blit(text, (10 * y_scale, 0))
    screen.blit(text2, (10 * y_scale, 24 * y_scale))
    screen.blit(text3, (WIDTH - text_width - 10 * y_scale, 0))
    screen.blit(text4, (WIDTH - text_width - 10 * y_scale, 24 * y_scale))
    pygame.display.update()


def new_game():
    player = Player()
    platforms = [Platform(HEIGHT - 1, 0)]
    platforms[0].x = 0
    platforms[0].width = WIDTH
    platforms[0].color = (0, 0, 0)
    springs = []
    time_scale = 1
    prev_time = pygame.time.get_ticks()
    return player, platforms, springs, time_scale, prev_time

def is_game_over(player):
    if player.score == 0 and player.y + Player.height > HEIGHT - 2:
        player.y = HEIGHT - 2 - Player.height
        player.jump()
        return False
    elif player.y > HEIGHT:
        return True
    return False


    
    


def read_high_score():
    if os.path.isfile(file_name) == False:
        file = open(file_name, "w")
        file.write("0")
        file.close()
    file = open(file_name, "r+")
    high_score = int(file.read())
    file.close()
    return high_score


class DoodleJumpEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def get_scores(self):
        return self.scores
    def __init__(self, render_mode=None):
        
        # initialize all the doodlejump game objects
        self.player, self.platforms, self.springs, self.time_scale, self.prev_time = new_game()
        self.high_score = read_high_score()
        self.num_deaths = 0
        self.scores = []
        self.start_time = 0
        self.end_time = 0
        self.runtimes = []
        self.score_per_moves = []
        self.moves_performed = 0
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                #  TODO fix observation space for PPO
                "agent": spaces.Box(np.array([0,0]), np.array([WIDTH,HEIGHT]), dtype=float),
                "target_platform": spaces.Box(np.array([0,0]), np.array([WIDTH,HEIGHT]), dtype=float),
                "target_spring": spaces.Box(np.array([0,0]), np.array([WIDTH,HEIGHT]), dtype=float),
            }
        )

        # # We have 3 actions, corresponding to "right", "left", "still"
        self.action_space = spaces.Discrete(3)

        # for rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    
    def simulate(self, action):


        distances = [99999]*len(self.platforms)
        for i in range(0,len(distances)):
            if (self.platforms[i].y > self.player.y): continue
            distances[i] = abs(self.player.x-(self.platforms[i].x-self.platforms[i].width/2)) + abs(self.player.y-self.platforms[i].y)
        pos = np.argmin(distances, axis = 0)
        closest = self.platforms[pos]
        toLeft = False
        toRight = False
        if (closest.x > self.player.x): toRight = True
        elif (closest.x < self.player.x): toLeft = True

        # triggered when the doodle dies
        if is_game_over(self.player):
            self.end_time = time.time()
            self.scores.append(self.player.score)
            self.runtimes.append(self.end_time-self.start_time)
            self.score_per_moves.append(self.player.score/self.moves_performed)
            self.player, self.platforms, self.springs, self.time_scale, self.prev_time = new_game()
            self.num_deaths += 1
            # file = open("scores2.txt", 'a')
            # file.write(str(sum(self.scores)/len(self.scores)) + '\n')
            # file.close()
            # file = open("timePerRun.txt", 'a')
            # file.write(str(sum(self.runtimes)/len(self.runtimes)) + '\n')
            # file.close()
            file = open("scorePerMove2.txt", 'a')
            file.write(str(sum(self.score_per_moves)/len(self.score_per_moves)) + '\n')
            file.close()
            self.start_time = time.time()
            self.moves_performed = 0
            #  TODO add reset function to here if we want it to call env.reset() every time the doodle dies
        # update high score
        if self.player.score > self.high_score:
            self.high_score = self.player.score
            file = open(file_name, "w")
            file.write(str(int(self.high_score)))
            file.close()
        # necessary for updating the high score as well as rendering the screen
        get_event(self.high_score)
        # move doodle based on input action (0,1,2) 0 = left, 1  = still, 2 = right
        if (not toLeft and  not toRight) :self.player.move(1, self.time_scale)
        elif (toLeft and not toRight): self.player.move(0, self.time_scale)
        elif (toRight and not toLeft): self.player.move(2, self.time_scale)
        self.moves_performed+=1
        # check if player go above half of screen's height
        if self.player.y < HEIGHT // 2 - self.player.height:
            movement = HEIGHT // 2 - self.player.height - self.player.y
            self.player.y = HEIGHT // 2 - self.player.height
        else:
            movement = 0
        self.player.score += movement / 4 / y_scale
        update_game(self.player, self.platforms, self.springs, self.time_scale, movement)

        if RENDER:
            render_game(screen, self.player, self.platforms, self.springs, self.time_scale, self.high_score)

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

        # deleat platforms below the screen
        i = 0
        while i < len(self.platforms):
            if self.platforms[i].y > HEIGHT:
                del self.platforms[i]
            i += 1
        i = 0

        # deleat springs below the screen
        while i < len(self.springs):
            if self.springs[i].y > HEIGHT:
                del self.springs[i]
            i += 1

        # generate platforms&springs
        while self.platforms[-1].y + self.platforms[-1].height >= 0:
            gap = random.randint(gap_lower_bound, gap_upper_bound) * y_scale
            platform = Platform(self.platforms[-1].y - gap, player.score)

            # can't have 3 fake platforms in a row
            if not (platform.type == 2 and self.platforms[-1].type == 2 and self.platforms[-2].type == 2):
                self.platforms.append(platform)
            # draw a spring if the platform have it
            if platform.have_spring:
                self.springs.append(Spring(platform))
    
    

    def _get_obs(self):
        return {"agent": self._agent_location,
                "target_platform": self._platform_location,
                "target_spring": [10,10]
                # "target"
                # TODO make "target" be nearest valid platform
                # , "target": self._target_location
                }
    

    def _get_info(self):
        # TODO implement distance to nearest platform alg
        distances = [99999]*len(self.platforms)
        # get coords of player
        x = self.player.x
        y = self.player.y
        for i in range(0,len(distances)-1): # for each platform
            if (self.platforms[i].y > y): continue
            distances[i] = abs(x-(self.platforms[i].x+self.platforms[i].width/2)) + abs(y-self.platforms[i].y)
        pos = np.argmin(distances, axis=0)
        self._platform_location = [self.platforms[pos].x,self.platforms[pos].y]
        return {"nearest": (self.platforms[pos].x, self.platforms[pos].y)}
        # return {"distance": 5}
        # return None



    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # print("Selman ERIS")
        super().reset(seed=seed)

        # Choose the agent's location 
        self._agent_location = [(WIDTH - 32 * y_scale) // 2, HEIGHT - 32 * y_scale]

        info = self._get_info()
        
        observation = self._get_obs()
        

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    def get_rewards(self,score,prev_score):
        if(debug): ic(round(score))
        if prev_score<score: return 1
        else: return 0
    def step(self, action):
        # TODO implement termination requirements (right now automatically false)
        # ideas: terminate after a certain height is reached, terminate after a certain reward threshhold, 
        #        terminate whenever we want to update weights 
        terminated = False
        if(self.num_deaths == 1):
            terminated = True
            self.num_deaths = 0
            if(debug):ic('Dead')
            # time.sleep(5)
        # terminated = np.array_equal(self._agent_location, self._target_location)
        
        
        info = self._get_info()
        observation = self._get_obs()
        prev_score=self.player.score
        
        self.new_platforms(self.player)
        self.simulate( action)
        reward = self.get_rewards(self.player.score,prev_score)
        

         # Prevent the code from running too fast during a simulation
        # if not RENDER: time.sleep(0.01)
        
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    