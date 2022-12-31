import numpy as np

class QAnalyzer():
    def __init__(self, QList, Name_List):
        self.Q_List = QList
        self.Name_List = Name_List

    def Analyze_Actions(self):
        for Q_Pair, name in zip(self.Q_List, self.Name_List):
            print(name)
            for Q in Q_Pair:
                for i in range(len(Q)):
                    for j in range(len(Q[i])):
                        for k in range(len(Q[i][j])):
                            if Q[i][j][k].ndim == 1:
                                    if not np.array_equal(Q[i][j][k], np.zeros(5)):
                                        print(Q[i][j][k])
                            else:
                                for q_values in Q[i][j][k]:
                                    if not np.array_equal(q_values, np.ones(5)) and not np.array_equal(q_values, np.zeros(5)):
                                        print(q_values)