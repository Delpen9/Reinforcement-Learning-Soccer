from QLearning import QLearningAgent
from FriendQ import FriendQAgent
from FoeQ import FoeQAgent
from CEQ import CEQAgent
from Grapher import Grapher
from QAnalyzer import QAnalyzer
import numpy as np
import pickle

def main():
    ## CONFIG SETTINGS
    ## -----------------
    SAVE_AGENTS = True
    LOAD_AGENTS = not SAVE_AGENTS
    LOAD_GRAPHS = True
    ANALYZE_Q_VALUES = True
    ## -----------------

    if SAVE_AGENTS == True:
        agent_one = QLearningAgent()
        agent_one.Run()
        with open('QLearner.pickle', 'wb') as handle:
            pickle.dump(agent_one, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        agent_two = FriendQAgent()
        agent_two.Run()
        with open('FriendQ.pickle', 'wb') as handle:
            pickle.dump(agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        agent_three = FoeQAgent()
        agent_three.Run()
        with open('FoeQ.pickle', 'wb') as handle:
            pickle.dump(agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        agent_four = CEQAgent()
        agent_four.Run()
        with open('CEQ.pickle', 'wb') as handle:
            pickle.dump(agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    elif LOAD_AGENTS == True:
        with open('QLearner.pickle', 'rb') as handle:
            agent_one = pickle.load(handle)
            handle.close()

        with open('FriendQ.pickle', 'rb') as handle:
            agent_two = pickle.load(handle)
            handle.close()

        with open('FoeQ.pickle', 'rb') as handle:
            agent_three = pickle.load(handle)
            handle.close()

        with open('CEQ.pickle', 'rb') as handle:
            agent_four = pickle.load(handle)
            handle.close()

    if LOAD_GRAPHS == True:
        agent_one_y_axis = agent_one.errors
        agent_one_x_axis = np.arange(1, len(agent_one_y_axis) + 1)
        grapher_one = Grapher(agent_one_x_axis, 'Simulation Iteration', agent_one_y_axis, 'Q-value Difference', 'Q-learner', 0.5)
        grapher_one.Graph()

        agent_two_y_axis = agent_two.errors
        agent_two_x_axis = np.arange(1, len(agent_two_y_axis) + 1)
        grapher_two = Grapher(agent_two_x_axis, 'Simulation Iteration', agent_two_y_axis, 'Q-value Difference', 'Friend-Q', 15.0)
        grapher_two.Graph()

        agent_three_y_axis = agent_three.errors
        agent_three_x_axis = np.arange(1, len(agent_three_y_axis) + 1)
        grapher_three = Grapher(agent_three_x_axis, 'Simulation Iteration', agent_three_y_axis, 'Q-value Difference', 'Foe-Q', 50.0)
        grapher_three.Graph()

        agent_four_y_axis = agent_four.errors
        agent_four_x_axis = np.arange(1, len(agent_four_y_axis) + 1)
        grapher_four = Grapher(agent_four_x_axis, 'Simulation Iteration', agent_four_y_axis, 'Q-value Difference', 'Correlated-Q', 50.0)
        grapher_four.Graph()

    if ANALYZE_Q_VALUES == True:
        Q_Values = [
            [agent_one.Q_1, agent_one.Q_2],
            [agent_two.Q_1, agent_two.Q_2],
            [agent_three.Q_1, agent_three.Q_2],
            [agent_four.Q_1, agent_four.Q_2]
        ]
        name_list = ['Q_Learner', 'Friend_Q', 'Foe_Q', 'Correlated_Q']
        q_analyzer = QAnalyzer(Q_Values, name_list)
        q_analyzer.Analyze_Actions()

if __name__ == "__main__":
    main()