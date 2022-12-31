from Environment import Environment
import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class CEQAgent():
    def __init__(self):
        self.no_of_steps = 10000
        self.Q_1 = 1 + np.zeros((8, 8, 2, 5, 5))
        self.Q_2 = 1 + np.zeros((8, 8, 2, 5, 5))
        self.Pi = 0.04 + np.zeros((8, 8, 2, 5, 5))
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

    def choose_action(self):
        if np.random.random() < self.epsilon:
            index = np.random.randint(0, 25)
            action_0 = index // 5
            action_1 = index - 5 * action_0
            return np.array([action_0, action_1])
        else:
            indices = np.arange(0, 25)
            probabilities = self.Pi[self.state[0]][self.state[1]][self.state[2]].reshape(25)
            random_value = np.random.random(1)[0]

            choice_value = 0
            for index, probability in zip(indices, probabilities):
                choice_value += probability
                if choice_value > random_value:
                    action_0 = index // 5
                    action_1 = index - 5 * action_0
                    return np.array([action_0, action_1])

    def create_diagonal_matrix_from_matrices(self, Q_val):
        concat_one = np.zeros((len(Q_val), len(Q_val[0]) * 4))
        matrix_one = np.hstack((Q_val - Q_val[0], concat_one))

        concat_one = np.zeros((len(Q_val), len(Q_val[0]) * 1))
        concat_two = np.zeros((len(Q_val), len(Q_val[0]) * 3))
        matrix_two = np.hstack((concat_one, Q_val - Q_val[1], concat_two))

        concat_one = np.zeros((len(Q_val), len(Q_val[0]) * 2))
        concat_two = np.zeros((len(Q_val), len(Q_val[0]) * 2))
        matrix_three = np.hstack((concat_one, Q_val - Q_val[2], concat_two))

        concat_one = np.zeros((len(Q_val), len(Q_val[0]) * 3))
        concat_two = np.zeros((len(Q_val), len(Q_val[0]) * 1))
        matrix_four = np.hstack((concat_one, Q_val - Q_val[3], concat_two))

        concat_one = np.zeros((len(Q_val), len(Q_val[0]) * 4))
        matrix_five = np.hstack((concat_one, Q_val - Q_val[4]))

        diagonal_matrix = np.concatenate((matrix_one, matrix_two, matrix_three, matrix_four, matrix_five), axis = 0)
        return diagonal_matrix

    def linear_program(self):
        try:
            Q_val = self.Q_1[self.state[0]][self.state[1]][self.state[2]]
            diagonal_matrix = self.create_diagonal_matrix_from_matrices(Q_val)
            matrix_one = diagonal_matrix[np.arange(1, 24), :]

            Q_val = self.Q_2[self.state[0]][self.state[1]][self.state[2]]
            diagonal_matrix = self.create_diagonal_matrix_from_matrices(Q_val)
            column_indices = np.concatenate((np.arange(0, 21, 5), np.arange(1, 22, 5), np.arange(2, 23, 5), np.arange(3, 24, 5), np.arange(4, 25, 5)), axis = 0)
            matrix_two = diagonal_matrix[np.arange(1, 24)][:][column_indices]

            c = matrix((self.Q_1[self.state[0]][self.state[1]][self.state[2]] + self.Q_2[self.state[0]][self.state[1]][self.state[2]].T).reshape(25))
            
            identity_matrix = -1 + np.identity(25)
            combined_matrix = np.concatenate((matrix_one, matrix_two), axis = 0)
            G_matrix = np.concatenate((combined_matrix, identity_matrix), axis = 0)
            G = matrix(G_matrix)

            h_matrix = np.zeros(65)
            h = matrix(h_matrix)

            A_matrix = 1 + np.zeros((1, 25))
            A = matrix(A_matrix)

            b = matrix(1.0)

            sol = solvers.lp(c=c, G=G, h=h, A=A, b=b)
            if sol['x'] is None:
                raise Exception('No solution found.')

            array = np.abs(sol['x'])
            probability = np.abs(array.reshape((5, 5))) / np.sum(array, axis = 0)
            V_1_Update = np.sum(self.Q_1[self.state[0]][self.state[1]][self.state[2]] * probability, axis = None)
            V_2_Update = np.sum(self.Q_2[self.state[0]][self.state[1]][self.state[2]].T * probability, axis = None)

        except:
            return None, None, None

        return probability, V_1_Update, V_2_Update

    def Run(self):
        i = 0
        while i < self.no_of_steps:
            environment = Environment()
            self.state = environment.state
            j = 0
            while(True):
                value_old = self.Q_1[2][1][1][2][4]
                ## print(value_old)

                i, j = i + 1, j + 1
                self.epsilon = (10**(np.log10(self.epsilon_min)/self.no_of_steps)) ** i

                actions = self.choose_action()

                state_new, rewards, done = environment.step(actions)

                self.alpha = (10**(np.log10(self.alpha_min)/self.no_of_steps)) ** i

                self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[0]][actions[1]] = (1 - self.alpha) * self.Q_1[self.state[0]][self.state[1]][self.state[2]][actions[0]][actions[1]] + self.alpha * (rewards[0] + self.gamma * self.V_1[state_new[0]][state_new[1]][state_new[2]])
                
                self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[1]][actions[0]] = (1 - self.alpha) * self.Q_2[self.state[0]][self.state[1]][self.state[2]][actions[1]][actions[0]] + self.alpha * (rewards[1] + self.gamma * self.V_1[state_new[0]][state_new[1]][state_new[2]].T)

                probability, v_1_new, v_2_new = self.linear_program()
                
                if probability != None:
                    self.Pi[self.state[0]][self.state[1]][self.state[2]] = probability
                    self.V_1[self.state[0]][self.state[1]][self.state[2]] = v_1_new
                    self.V_2[self.state[0]][self.state[1]][self.state[2]] = v_2_new
                
                self.state = state_new

                value_new = self.Q_1[2][1][1][2][4]
                ## print(value_new)

                self.errors.append(abs(value_new - value_old))

                if done == True or j <= 100:
                    break