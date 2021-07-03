import numpy as np
import tensorflow as tf

from tensorflow import keras
from collections import deque

# TODO: add customization of DDQN (if using it)
# Contains all information with respect to an agent in a simulation
class DQNAgent:
        # Skeleton is the skeleton of the model
        def __init__(self, skeleton, policy, outputs, gamma):
                # Using a DDQN (TODO: ?)
                self.target = keras.models.clone_model(skeleton)
                self.main = keras.models.clone_model(skeleton)

                self.target.set_weights(skeleton.get_weights())
                self.main.set_weights(skeleton.get_weights())

                # Setting policy
                self.policy = policy

                # Replay buffer
                self.rbf = deque(maxlen = 2000)

                # Misc
                self.nouts = outputs
                self.episode = 0
                self.gamma = gamma
                self.loss = keras.losses.mean_squared_error # keras.losses.Huber()
                self.optimizer = keras.optimizers.Adam(lr = 1e-2) # 6e-3)
        
        # Policy
        def do_policy(self, state):
                sout = self.policy(self.nouts, state)

                # If the policy did not activate its secondary function
                if sout == -1:
                        Q_values = self.main.predict(state[np.newaxis])
                        return np.argmax(Q_values[0])
                
                # Value of secondary function
                return sout
        
        # Playing a step
        def step(self, env, state):
            # Transfer main weights to target weights every now and then
            # if epsilon % 50 == 0:
            #    self.target.set_weights(self.main.get_weights())

            action = self.do_policy(state)
            nstate, reward, done, info = env.step(action)
            self.rbf.append((state, action, reward, nstate, done))
            return nstate, reward, done, env
        
        # Sampling experiences
        def sample(self, size):
                indices = np.random.randint(len(self.rbf), size = size)
                batch = [self.rbf[index] for index in indices]
                
                states, actions, rewards, nstates, dones = [
                        np.array(
                                [experience[field_index] for experience in batch]
                        )
                for field_index in range(5)]

                return states, actions, rewards, nstates, dones
        
        # Training for each step
        def train(self, size):
            states, actions, rewards, nstates, dones = self.sample(size)

            next_Qvs = self.main.predict(nstates)
            max_next_Qvs = np.max(next_Qvs, axis = 1)
            
            tgt_Qvs = (rewards + (1 - dones) * self.gamma * max_next_Qvs)
            tgt_Qvs = tgt_Qvs.reshape(-1, 1)

            mask = tf.one_hot(actions, self.nouts)
            with tf.GradientTape() as tape:
                all_Qvs = self.main(states)
                Qvs = tf.reduce_sum(all_Qvs * mask, axis = 1, keepdims = True)
                loss = tf.reduce_mean(self.loss(tgt_Qvs, Qvs))

            grads = tape.gradient(loss, self.main.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.main.trainable_variables))

            '''
            states, actions, rewards, nstates, dones = self.sample(size)

            next_Qvs = self.main.predict(nstates)
            best_nactions = np.argmax(next_Qvs, axis=1)
            next_mask = tf.one_hot(best_nactions, self.nouts).numpy()
            next_best_Qvs = (self.target.predict(nstates) * next_mask).sum(axis = 1)
            
            target_Qvs = (rewards + (1 - dones) * self.gamma * next_best_Qvs)
            target_Qvs = target_Qvs.reshape(-1, 1)

            mask = tf.one_hot(actions, self.nouts)
            with tf.GradientTape() as tape:
                all_Qvs = self.main(states)
                Qvs = tf.reduce_sum(all_Qvs * mask, axis=1, keepdims=True)
                loss = tf.reduce_mean(self.loss(target_Qvs, Qvs))

            grads = tape.gradient(loss, self.main.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.main.trainable_variables))
            '''