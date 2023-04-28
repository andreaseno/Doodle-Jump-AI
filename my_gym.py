#ex.py

import gym
import gym_doodlejump
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.python.keras.losses as kls
import numpy as np
from tensorflow import keras
from icecream import ic

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flat = keras.layers.Flatten(input_shape=(6,))
        self.d1 = tf.keras.layers.Dense(6,activation='relu')
        self.v = tf.keras.layers.Dense(1, activation = None)

    def call(self, input_data):
        input_data = convert_state(input_data)
        flat = self.flat(input_data)
        d1 = self.d1(flat)
        v = self.v(d1)
        return v
    

class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flat = keras.layers.Flatten(input_shape=(6,))
        self.d1 = tf.keras.layers.Dense(6,activation='relu')
        self.a = tf.keras.layers.Dense(2,activation='softmax')

    def call(self, input_data):
        input_data = convert_state(input_data)
        flat = self.flat(input_data)
        d1 = self.d1(flat)
        a = self.a(d1)
        return a
    
class agent():
    def __init__(self):
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()
        self.clip_pram = 0.2

          
    def act(self,state):
        prob = self.actor(state)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
   
    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op, a  in zip(probability, adv, old_probs, actions):
                        t =  tf.constant(t)
                        #op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb[a],op[a])
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss

    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),2))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            # TODO: MODIFY THIS LINE
            state = states[-1]
            p = self.actor(state, training=True)
            v =  self.critic(state,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss
def test_reward(env):
    total_reward = 0
    state, info = env.reset()
    done = False
    while not done:
        action = np.argmax(agentoo7.actor(state).numpy())
        next_state, reward, done, _, info = env.step(action)
        state = next_state
        total_reward += reward

    return total_reward

def preprocess1(states, actions, rewards, done, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv    


env = gym.make('doodlejump-v0', render_mode='human')

observation, info = env.reset()
action = env.action_space.sample()

tf.random.set_seed(336699)
agentoo7 = agent()
steps = 5000
ep_reward = []
total_avgr = []
target = False 
best_reward = 0
avg_rewards_list = []


def convert_state(state):
    ic(state)
    statelist = [*state["agent"],*state["target_platform"],*state["target_spring"]]
    statelist = np.array(statelist)
    statelist = statelist.reshape(1, -1)  # reshape to (1, 6)
    return statelist

for s in range(steps):
    if target == True:
        break
    
    done = False
    state, info = env.reset()
    print("LEO PHAN")
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []
    probs = []
    dones = []
    values = []
    print("new episod")

    for e in range(128):
    
        action = agentoo7.act(state)
        value = agentoo7.critic(state).numpy()
        next_state, reward, done, _, info = env.step(action)
        dones.append(1-done)
        rewards.append(reward)
        states.append(state)
        #actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
        actions.append(action)
        prob = agentoo7.actor(state)
        probs.append(prob[0])
        values.append(value[0][0])
        state = next_state
        if done:
            env.reset()
    
    value = agentoo7.critic(state).numpy()
    values.append(value[0][0])
    np.reshape(probs, (len(probs),2))
    probs = np.stack(probs, axis=0)

    states, actions,returns, adv  = preprocess1(states, actions, rewards, dones, values, 1)

    for epocs in range(10):
        al,cl = agentoo7.learn(states, actions, adv, probs, returns)
        # print(f"al{al}") 
        # print(f"cl{cl}")   

    avg_reward = np.mean([test_reward(env) for _ in range(5)])
    print(f"total test reward is {avg_reward}")
    avg_rewards_list.append(avg_reward)
    if avg_reward > best_reward:
            print('best reward=' + str(avg_reward))
            agentoo7.actor.save('model_actor_{}_{}'.format(s, avg_reward), save_format="tf")
            agentoo7.critic.save('model_critic_{}_{}'.format(s, avg_reward), save_format="tf")
            best_reward = avg_reward
    if best_reward == 200:
            target = True
    env.reset()

env.close()




# for i in range(10000):
#     if i % 10 == 0:
#         action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
