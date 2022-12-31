from Environment import Environment
import numpy as np

class QLearningAgent():
    def __init__(self):
        self.no_of_steps = 2000
        self.Q_1 = np.zeros((8, 8, 2, 5))
        self.Q_2 = np.zeros((8, 8, 2, 5))
        self.state = np.array([])
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99995
        self.gamma = 0.9
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.99951
        self.errors = []
        self.random_seed = np.random.seed(9999)
    
    def choose_action(self, Q):
        if np.random.random(1)[0] < self.epsilon:
            action = np.random.randint(0, 5)
            return action
        else:
            max_value = np.max(Q[self.state[0]][self.state[1]][self.state[2]])
            for index, value in enumerate(Q[self.state[0]][self.state[1]][self.state[2]], 0):
                if value == max_value:
                    action = index
                    return action

    def Run(self):
        for i in range(self.no_of_steps):
            environment = Environment()
            self.state = environment.state

            while(True):
                value_old = self.Q_1[2][1][1][2]

                actions = [self.choose_action(self.Q_1), self.choose_action(self.Q_2)]

                state_new, rewards, done = environment.step(actions)

                player_1_memory = self.gamma * max(self.Q_1[state_new[0]][state_new[1]][state_new[2]]) if done != True else 0
                player_2_memory = self.gamma * max(self.Q_2[state_new[0]][state_new[1]][state_new[2]]) if done != True else 0

                self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[0]] = self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[0]] + self.alpha * (rewards[0] + player_1_memory - self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[0]])
                self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[1]] = self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[1]] + self.alpha * (rewards[0] + player_2_memory - self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[1]])

                value_new = self.Q_1[2][1][1][2]

                self.errors.append(abs(value_new - value_old))
                
                if done != True:
                    self.epsilon = self.epsilon_decay * self.epsilon if self.epsilon_decay * self.epsilon > self.epsilon_min else self.epsilon_min
                    self.alpha = self.alpha_decay * self.alpha if self.alpha_decay * self.alpha > self.alpha_min else self.alpha_min
                    self.state = state_new
                else:
                    break