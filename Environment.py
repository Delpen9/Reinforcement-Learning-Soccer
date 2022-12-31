import numpy as np

class Environment:
    def __init__(self):
        self.player_one_position = np.array([0, 2])
        self.player_two_position = np.array([0, 1])
        self.player_one_position_new = self.player_one_position.copy()
        self.player_two_position_new = self.player_two_position.copy()
        self.ball = 1
        self.goal = np.array([0, 3])
        self.actions = np.array([[0, 0], [0, 1], [1, 0], [-1, 0], [0, -1]])
        self.rewards = np.array([0, 0])
        self.done = False
        self.state = np.array([self.player_one_position[0] * 4 + self.player_one_position[1], self.player_two_position[0] * 4 + self.player_two_position[1], self.ball])

    def player_movement(self, movement_order, actions):
        moving_player_position = self.player_one_position.copy() if movement_order == 0 else self.player_two_position.copy()
        moving_player_position_new = self.player_one_position_new.copy() if movement_order == 0 else self.player_one_position_new.copy()
        stationary_player_position = self.player_two_position.copy() if movement_order == 0 else self.player_one_position.copy()
        stationary_player_position_new = self.player_two_position_new.copy() if movement_order == 0 else self.player_one_position_new.copy()

        moving_player_position_new = moving_player_position + self.actions[actions[movement_order]]

        if (any(True if a == b else False for a, b in zip(moving_player_position_new, stationary_player_position_new))):
            self.ball = 1 - movement_order if self.ball == movement_order else self.ball

        elif (moving_player_position_new[0] in [0, 1] and moving_player_position_new[1] in [0, 1, 2, 3]):
            moving_player_position = moving_player_position_new

            scored_for_self = moving_player_position[1] == self.goal[movement_order]
            scored_for_opponent = moving_player_position[1] == self.goal[1 - movement_order]
            if ((scored_for_self or scored_for_opponent) and self.ball == movement_order):
                self.rewards = np.array([-100, 100]) if movement_order == 0 else np.array([100, -100])
                self.rewards = self.rewards[::-1] if scored_for_opponent else self.rewards
                self.done = True
                if movement_order == 0:
                    self.state = np.array([moving_player_position[0] * 4 + moving_player_position[1], stationary_player_position[0] * 4 + stationary_player_position[1], self.ball])
                else:
                    self.state = np.array([stationary_player_position[0] * 4 + stationary_player_position[1], moving_player_position[0] * 4 + moving_player_position[1], self.ball])

        if movement_order == 0:
            self.player_one_position = moving_player_position.copy()
            self.player_one_position_new = moving_player_position_new.copy()
        else:
            self.player_two_position = moving_player_position.copy()
            self.player_two_position_new = moving_player_position_new.copy()

    def step(self, actions):
        self.rewards = np.array([0, 0])
        self.done = False

        ## First Mover
        ## ----------------
        movement_order = np.random.randint(2)
        self.player_movement(movement_order, actions)
        if self.done == True:
            return self.state, self.rewards, self.done
        ## ----------------

        ## Second Mover
        ## ----------------
        self.player_movement(1 - movement_order, actions)
        ## ----------------

        return self.state, self.rewards, self.done