from suites import *

# Epsilon greedy policy
def egreedy_policy(outputs, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(outputs)
    else:
        return -1

# Contains all information with respect to an agent in a simulation
class Agent:
        # Skeleton is the skeleton of the model
        def __init__(self, skeleton, policy, outputs, gamma):
                # Using a DDQN
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
        
        # Policy
        def do_policy(self, state, epsilon):
                sout = self.policy(self.nouts, epsilon)

                # If the policy did not activate its secondary function
                if sout == -1:
                        Q_values = self.main.predict(state[np.newaxis])
                        return np.argmax(Q_values[0])
                
                # Value of secondary function
                return sout
        
        # Playing a step
        def step(self, env, state, epsilon):
                action = self.do_policy(state, epsilon)
                nstate, reward, done, info = env.step(action)
                self.rbf.append((state, action, reward, nstate, done))
                return nstate, reward, done, info
        
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
                tgt_Qvs = target_Q_values.reshape(-1, 1)

                mask = tf.one_hot(actions, self.nouts)

                with tf.GradientTape() as tape:
                        all_Qvs = model(states)
                        Qvs = tf.reduce_sum(all_Qvs * mask, axis = 1, keepdims = True)
                        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
                
                grads = tape.gradient(loss, self.main.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.main.trainable_variables))

def run_experiment(env_name):
        env = gym.make(env_name)

        model_skeleton = models[env]

        # Simple example

for pair in models:
    run_experiment(pair)
