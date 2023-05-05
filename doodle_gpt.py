import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'True'

import pygame, sys, random, math, time, numpy as np

SEED_NUM = 2023
random.seed(SEED_NUM)

from icecream import ic

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

MAX_PLATFORMS = 24
doodle_min_x, doodle_max_x = 0, WIDTH
doodle_min_y, doodle_max_y = 0, HEIGHT
platform_min_x, platform_max_x = 0, WIDTH
platform_min_y, platform_max_y = 0, HEIGHT

low = np.array([doodle_min_x, doodle_min_y] + [platform_min_x, platform_min_y] * MAX_PLATFORMS)
high = np.array([doodle_max_x, doodle_max_y] + [platform_max_x, platform_max_y] * MAX_PLATFORMS)


REPEATED_MOVES = 10


import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
from gym import spaces

class DoodleJumpEnv(gym.Env):
    def __init__(self):
        super(DoodleJumpEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # left, right, do nothing
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        player, platforms, springs, time_scale, prev_time = self.new_game()
        self.player = player
        self.platforms = platforms
        self.time_scale = time_scale
        self.springs = springs
        self.prev_time = prev_time
        self.high_score = self.read_high_score()
        self.scores = []
        self.score_per_moves = []
        self.moves_performed = 0
        self.prev_score = 0
        
        self.new_platforms()
        
        self.max_consecutive_actions = 5
        self.consecutive_actions_count = 0
        self.prev_action = None
        

    def step(self, action):
        # Apply the action in the game
        self.apply_action(action)

        # Update the game state (e.g., Doodler position, platforms, etc.)
        self.update_game_state()

        self.new_platforms()

        # Calculate the reward based on the agent's performance
        reward = self.calculate_reward(action)

        # Determine if the game is over
        done = self.is_game_over()
        if done:
            self.scores.append(self.player.score)
            total = 0
            for score in self.scores:
                total += score
            file = open('ppo-scores.txt', "a")
            file.write(str(total/len(self.scores)) + '\n')
            file.close()

            self.score_per_moves.append(self.player.score/self.moves_performed)

            file = open("ppo-scoresPerMove.txt", 'a')
            file.write(str(sum(self.score_per_moves)/len(self.score_per_moves)) + '\n')
            file.close()

            self.moves_performed = 0

        # Get the new state
        state = self.get_state()
        
        if RENDER and not done:
            self.render()

        return state, reward, done
    
    
    def apply_action(self, action):
        r = action
        # mvoe left
        if   action == 0: move = (True, False)
        # move right
        elif action == 1: move = (False, True)
        # stand still
        else:             move = (False, False)
        
        self.player.move(*move,self.time_scale)
        self.moves_performed+=1
    
    def update_game_state(self):
        player = self.player
        
        if self.player.score > self.high_score:
            self.high_score = self.player.score
        
        # check if player go above half of screen's height
        if player.y < HEIGHT // 2 - player.height:
            movement = HEIGHT // 2 - player.height - player.y
            player.y = HEIGHT // 2 - player.height
        else:
            movement = 0
        player.score += movement / 4 / y_scale
        self.update_game(movement)
    
    
    def update_game(self, movement):
        player      = self.player
        platforms   = self.platforms
        springs     = self.springs
        time_scale  = self.time_scale
                
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
    
    
    def reset(self):
        random.seed(SEED_NUM)
        
        # Reset the game (e.g., set the doodle's position to the starting point)
        self.reset_game()

        # Reset the platforms (e.g., generate new platforms or reset their positions)
        self.reset_platforms()

        # Clear the state history if you're maintaining one
        if hasattr(self, 'state_history'):
            self.state_history.clear()

        # Get the initial state
        initial_state = self.get_state()

        # Add the initial state to the state history if you're maintaining one
        if hasattr(self, 'state_history'):
            self.state_history.append(initial_state)

        return initial_state

    def reset_game(self):
        player, platforms, springs, time_scale, prev_time = self.new_game()
        self.player = player
        self.platforms = platforms
        self.time_scale = time_scale
        self.springs = springs
        self.prev_time = prev_time
        self.new_platforms()
        self.prev_score = 0
        
        file = open(file_name, "w")
        file.write(str(int(self.high_score)))
        file.close()
        self.high_score = self.read_high_score()
    
    def reset_platforms(self):
        pass

    
    def render(self, mode='human'):
        screen.fill(background_color)

        for platform in self.platforms:
            platform.draw(self.time_scale)

        for spring in self.springs:
            spring.draw()

        self.player.draw()

        #print scores
        font = pygame.font.SysFont("Comic Sans MS", int(24 * y_scale))
        text = font.render("Score:", True, (0, 0, 0))
        text2 = font.render(str(int(self.player.score)), True, (0, 0, 0))
        text3 = font.render("Best:", True, (0, 0, 0))
        text4 = font.render(str(int(self.high_score)), True, (0, 0, 0))
        text_width = max(text3.get_width(), text4.get_width())
        screen.blit(text, (10 * y_scale, 0))
        screen.blit(text2, (10 * y_scale, 24 * y_scale))
        screen.blit(text3, (WIDTH - text_width - 10 * y_scale, 0))
        screen.blit(text4, (WIDTH - text_width - 10 * y_scale, 24 * y_scale))
        pygame.display.update()
        
        # Prevent the code from running too fast during a simulation
        self.time_scale = (pygame.time.get_ticks() - self.prev_time) / 10 * TIME_SPEED
        self.prev_time  = pygame.time.get_ticks()
        time.sleep(0.01)
    
    
    def calculate_reward(self, action):
        # Reward for vertical progress (higher jumps)
        reward = self.player.score - self.prev_score
        
        # Penalize for moving in one direction for too long
        if action == self.prev_action:
            self.consecutive_actions_count += 1
        else:
            self.consecutive_actions_count = 0

        if self.consecutive_actions_count >= self.max_consecutive_actions:
            reward -= 10

        self.prev_action = action
        
        if reward > 0:
            self.prev_score = self.player.score

        # Penalty for falling off the screen
        if self.is_game_over():
            reward -= 100

        # ic(
        #     self.player.score,
        #     self.high_score,
        #     round(reward,1),
        # )
        return reward
    
    
    def is_game_over(self):
        player = self.player
        if player.score == 0 and player.y + Player.height > HEIGHT - 2:
            player.y = HEIGHT - 2 - Player.height
            player.jump()
            return False
        elif player.y > HEIGHT:
            return True
        return False
    
    
    def get_state(self):
        # Get the doodle's position (x, y)
        doodle_x = self.player.x
        doodle_y = self.player.y

        # Get the positions of the platforms (x, y) as a flattened list
        platform_positions = []
        for i in range(MAX_PLATFORMS):
            if i < len(self.platforms):
                platform = self.platforms[i]
                platform_x = platform.x + (64 * y_scale)/2
                platform_y = platform.y
                platform_positions.extend([platform_x, platform_y])
            else:
                platform_positions.extend([WIDTH, HEIGHT])
        # Combine the doodle's position and platform positions into a single state vector
        state = np.array([doodle_x, doodle_y] + platform_positions)

        return state
    
    """Game Methods"""
    
    def new_game(self):
        player = Player()
        platforms = [Platform(HEIGHT - 1, 0)]
        platforms[0].x = 0
        platforms[0].width = WIDTH
        platforms[0].color = (0, 0, 0)
        springs = []
        time_scale = 1
        prev_time = pygame.time.get_ticks()
        return player, platforms, springs, time_scale, prev_time
    
    
    def new_platforms(self):
        
        player = self.player
        platforms = self.platforms
        springs = self.springs
        
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
    
                
    def read_high_score(self):
        if os.path.isfile(file_name) == False:
            file = open(file_name, "w")
            file.write("0")
            file.close()
        file = open(file_name, "r+")
        high_score = int(file.read())
        file.close()
        return high_score
    
    

class PPOAgent:
    def __init__(self, state_size, action_size, gamma, epsilon, lr):
        self.policy = self.build_policy(state_size, action_size)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def build_policy(self, state_size, action_size):
        policy = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(50, activation='relu'),
            # tf.keras.layers.Dense(25, activation='relu'),
            # tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])
        return policy

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        logits = self.policy(state)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.numpy()[0], dist.log_prob(action)

    def compute_returns(self, rewards):
        returns = []
        return_so_far = 0
        for r in reversed(rewards):
            return_so_far = r + self.gamma * return_so_far
            returns.append(return_so_far)
        return returns[::-1]
                    
                
    def update(self, states, actions, rewards, log_probs):
        returns = self.compute_returns(rewards)
        advantages = np.array(returns) - np.array(rewards)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        for state, action, old_log_prob, advantage in zip(states, actions, log_probs, advantages):
            state = np.expand_dims(state, axis=0)
            with tf.GradientTape() as tape:
                logits = self.policy(state)
                dist = tfp.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(action)
                ic(new_log_prob)

                ratio = tf.exp(new_log_prob - old_log_prob)
                loss = -tf.reduce_mean(tf.minimum(ratio * advantage, tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage))
                ic(loss)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
                
                
        # Save the model's weights to a file
    def save_model_weights(agent, file_path):
        agent.policy.save_weights(file_path)

    # Load the model's weights from a file
    def load_model_weights(agent, file_path):
        agent.policy.load_weights(file_path)
                
    
"""Game Objects"""

class Player():
    jump_force = 8 * y_scale
    max_x_speed = 16 * x_scale * (.25)
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
        if score < 20:
            # self.type = 0
            self.type = random.choice([0, 0, 0, 0, 0, 0, 1, 1])
        elif score < 50:
            self.type = random.choice([0, 0, 0, 0, 0, 0, 1, 1])
        elif score < 100:
            self.type = random.choice([0, 0, 0, 0, 1, 1, 1, 1])
        elif score < 200:
            self.type = random.choice([0, 0, 0, 1, 1, 1, 1, 2])
        elif score < 300:
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
        
        
"""Support Functions"""
            

def train(agent, env, episodes, timesteps):
    os.remove(file_name)
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        states, actions, rewards, log_probs = [], [], [], []

        # for t in range(timesteps):
        while True:
            if done:
                
                break

            action, log_prob = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            total_reward += reward
                    
        agent.update(states, actions, rewards, log_probs)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}", flush=True)
        
            
if __name__ == "__main__":
    """
    state_size = 2 (doodle's position) + 2 * N (platforms' positions)
    """
    state_size = 2 + 2 * MAX_PLATFORMS
    action_size = 3
    gamma = 0.99
    epsilon = 0.2
    lr = 0.001

    env = DoodleJumpEnv()
    agent = PPOAgent(state_size, action_size, gamma, epsilon, lr)

    episodes = 2000
    timesteps = 100

    train(agent, env, episodes, timesteps)