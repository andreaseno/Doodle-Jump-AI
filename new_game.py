import pygame, sys, random, math, os, time

RENDER = False
# default resolution: 360 x 640
RESOLUTION = WIDTH, HEIGHT = 360, 640
TITLE = "Doodle Jump"
TIME_SPEED = 1
RANDOM_AI = True

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
    max_x_speed = 16 * x_scale
    x_acceleration = 0.15 * x_scale
    color = (255, 255, 0)
    color2 = (0, 255, 255)
    height = 32 * y_scale
    width = 32 * y_scale
    moves = []
    moves_performed = 0

    def __init__(self):
        self.y = HEIGHT - self.height
        self.x = (WIDTH - self.width) // 2
        self.y_speed = -self.jump_force
        self.x_speed = 0
        self.direction = 0
        self.moving_direction = 0
        self.score = 0

        for _ in range(1000):
            choice = random.choice([-1,0,1])
            self.moves.extend([choice] * 5)

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
        if(RANDOM_AI):
            self.type = 0
        else:
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


def new_platforms(player):
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
    while i < len(platforms):
        if platforms[i].y > HEIGHT:
            del platforms[i]
        i += 1
    i = 0

    # deleat springs below the screen
    while i < len(springs):
        if springs[i].y > HEIGHT:
            del springs[i]
        i += 1

    # generate platforms&springs
    while platforms[-1].y + platforms[-1].height >= 0:
        gap = random.randint(gap_lower_bound, gap_upper_bound) * y_scale
        platform = Platform(platforms[-1].y - gap, player.score)

        # can't have 3 fake platforms in a row
        if not (platform.type == 2 and platforms[-1].type == 2 and platforms[-2].type == 2):
            platforms.append(platform)
        # draw a spring if the platform have it
        if platform.have_spring:
            springs.append(Spring(platform))


def game_over():
    pass


def get_event():
    if RENDER:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                file = open(file_name, "w")
                file.write(str(int(high_score)))
                file.close()
                pygame.quit()
                sys.exit()
    pressed = pygame.key.get_pressed()
    left_key_pressed = pressed[pygame.K_LEFT] or pressed[pygame.K_a]
    right_key_pressed = pressed[pygame.K_RIGHT] or pressed[pygame.K_d]
    return left_key_pressed, right_key_pressed


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


def render_game(screen, player, platforms, springs, time_scale):
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

def simulate(player, platforms, springs, time_scale):
    (left_key_pressed, right_key_pressed) = get_event()

    if(RANDOM_AI):
        current_move = player.moves[player.moves_performed]
        if(current_move == -1):
            left_key_pressed = 1
            right_key_pressed = 0
        elif(current_move == 1):
            left_key_pressed = 0
            right_key_pressed = 1
        else:
            left_key_pressed = 1
            right_key_pressed = 1
        player.moves_performed+=1
        # add new random moves if initial ones are used
        if(player.moves_performed == len(player.moves)):
            choice = random.choice([-1,0,1])
            player.moves.extend([choice] * 5)
    player.move(left_key_pressed, right_key_pressed, time_scale)

    # check if player go above half of screen's height
    if player.y < HEIGHT // 2 - player.height:
        movement = HEIGHT // 2 - player.height - player.y
        player.y = HEIGHT // 2 - player.height
    else:
        movement = 0
    player.score += movement / 4 / y_scale
    update_game(player, platforms, springs, time_scale, movement)

    if RENDER:
        render_game(screen, player, platforms, springs, time_scale)



player, platforms, springs, time_scale, prev_time = new_game()


def read_high_score():
    if os.path.isfile(file_name) == False:
        file = open(file_name, "w")
        file.write("0")
        file.close()
    file = open(file_name, "r+")
    high_score = int(file.read())
    file.close()
    return high_score


high_score = read_high_score()
scores = []
avg_scores_times = []
time_per_run = []
score_per_move = []
start_time = time.time()
while True:
    simulate(player, platforms, springs, time_scale)
    if player.score > high_score:
        high_score = player.score

    new_platforms(player)

    if is_game_over(player):
        end_time = time.time()
        scores.append(player.score)
        score_per_move.append(player.score/player.moves_performed)
        time_per_run.append(end_time-start_time)
        total = 0
        total_time = 0
        total_moves = 0
        for score in scores:
            total += score
        for t in time_per_run:
            total_time += t
        for accuracy in score_per_move:
            total_moves += accuracy
        avg_scores_times.append(total/len(scores))
        # file = open('scores.txt', "a")
        # file.write(str(total/len(scores)) + '\n')
        # file.close()
        # file = open('timePerRun.txt', "a")
        # file.write(str(total_time/len(time_per_run)) + '\n')
        # file.close()
        file = open('scorePerMove.txt', "a")
        file.write(str(total_moves/len(score_per_move)) + '\n')
        file.close()

        player, platforms, springs, time_scale, prev_time = new_game()
        start_time = time.time()

    # Prevent the code from running too fast during a simulation
    if not RENDER: time.sleep(0.01)

    time_scale = (pygame.time.get_ticks() - prev_time) / 10 * TIME_SPEED
    prev_time = pygame.time.get_ticks()