from Environment import Environment
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class FoeQAgent():
    def __init__(self):
        self.no_of_steps = 10000
        self.Q_1 = 1 + np.zeros((8, 8, 2, 5, 5))
        self.Q_2 = 1 + np.zeros((8, 8, 2, 5, 5))
        self.Pi_1 = 0.2 + np.zeros((8, 8, 2, 5))
        self.Pi_2 = 0.2 + np.zeros((8, 8, 2, 5))
        self.V_1 = 1 + np.zeros((8, 8, 2))
        self.V_2 = 1 + np.zeros((8, 8, 2))
        self.state = np.array([])
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.gamma = 0.9
        self.alpha = 1.0
        self.alpha_min = 0.001
        self.errors = []
        self.random_seed = np.random.seed(9999)

    def choose_action(self, pi):
        if np.random.random(1)[0] < self.epsilon:
            action = np.random.randint(0, 5)
        else:
            indices = np.arange(0, 5)
            probabilities = pi[self.state[0]][self.state[1]][self.state[2]].reshape(5)
            random_value = np.random.random(1)[0]

            choice_value = 0
            for index, probability in zip(indices, probabilities):
                choice_value += probability
                if choice_value > random_value:
                    action = index
                    break
        return action
    
    def linear_program(self, Q):
        array = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        c = matrix(array)

        array_one = np.array([1.0, 1.0, 1.0, 1.0, 1.0]).reshape((5, 1))
        array_two = -1 + Q[self.state[0]][self.state[1]][self.state[2]]
        array_combined_one = np.concatenate((array_one, array_two), axis = 1)
        array_four = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape((5, 1))
        array_five = -1 + np.identity(5)
        array_combined_two = np.concatenate((array_four, array_five), axis = 1)
        array_combined = np.concatenate((array_combined_one, array_combined_two), axis = 0)
        G = matrix(array_combined)
        
        array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        h = matrix(array)
        
        array = [[0.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        A = matrix(array)

        b = matrix(1.0)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)

        array = np.abs(sol['x'])[1:]
        Q_Update = array.reshape((5)) / np.sum(array, axis = 0)
        V_Update = np.array(sol['x'][0])
        return Q_Update, V_Update

    def Run(self):
        i = 0
        while i < self.no_of_steps:
            environment = Environment()
            self.state = environment.state

            while(True):
                value_old = self.Q_1[2][1][1][4][2]
                ## print(value_old)

                i += 1
                self.epsilon = (10**(np.log10(self.epsilon_min)/self.no_of_steps)) ** i

                actions = [self.choose_action(self.Pi_1), self.choose_action(self.Pi_2)]

                state_new, rewards, done = environment.step(actions)

                self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[1]][actions[0]] = (1 - self.alpha) * self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[1]][actions[0]] + self.alpha * (rewards[0] + self.gamma * self.V_1[state_new[0]][state_new[1]][state_new[2]])
                
                ## Solve linear program
                self.Pi_1[self.state[0]][self.state[1]][self.state[2]], self.V_1[self.state[0]][self.state[1]][self.state[2]] = self.linear_program(self.Q_1)

                self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[0]][actions[1]] = (1 - self.alpha) * self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[0]][actions[1]] + self.alpha * (rewards[1] + self.gamma * self.V_2[state_new[0]][state_new[1]][state_new[2]])

                ## Solve linear program
                self.Pi_2[self.state[0]][self.state[1]][self.state[2]], self.V_2[self.state[0]][self.state[1]][self.state[2]] = self.linear_program(self.Q_2)

                value_new = self.Q_1[2][1][1][4][2]
                ## print(value_new)

                self.errors.append(abs(value_new - value_old))
                self.alpha = (10**(np.log10(self.alpha_min)/self.no_of_steps)) ** i

                if done == True or i > 1000:
                    break