import gym
from gym.spaces import Box
from gym_backgammon.envs.backgammon import Backgammon as Game, WHITE, BLACK, COLORS
from gym_backgammon.envs.rendering import Viewer
import numpy as np

STATE_W = 96
STATE_H = 96

SCREEN_W = 600
SCREEN_H = 500


class BackgammonEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array', 'state_pixels'],
                'render_fps': 4}

    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.game = Game()
        self.current_agent = None

        low = np.zeros(198)
        high = np.ones(198)

        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5

        state_space = Box(low=np.float32(low), high=np.float32(high), dtype=np.float32)

        self.observation_space = state_space
        self.action_space = state_space 
        self.counter = 0
        self.max_length_episode = 10000
        self.viewer = None


    def step(self, action):

        self.game.execute_play(self.current_agent, action)
        
        # get the board representation from the opponent player perspective (the current player has already performed the move)
        observation = self._get_obs()

        reward = 0
        done = False

        winner = self.game.get_winner()

        if winner is not None:
            # practical-issues-in-temporal-difference-learning, pag.3
            # ...leading to a final reward signal z. In the simplest case, z = 1 if White wins and z = 0 if Black wins
            if winner == WHITE:
                reward = 1
            # game terminated
            terminated, truncated = True, False
            return observation, reward, terminated, truncated, {"winner": winner}

        self.counter += 1

        if self.counter > self.max_length_episode:
            # game truncated
            terminated, truncated = False, True
            return observation, reward, terminated, truncated, {"winner": None}
        
        # normal step
        terminated, truncated = False, False
        return observation, reward, terminated, truncated, {"winner": None}

        

    def _get_obs(self):
        # returns the current representation of the board.
        return np.array(self.game.get_board_features(self.game.get_opponent(self.current_agent)) , dtype=np.float32)


    def _get_info(self):
        return {"agent_color": self.current_agent, "roll": self.roll}


    def reset(self, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset()
        
        # returns a two random integers at a range [start, end]
        def double_dice(start, end):
            return self.np_random.integers(start,end + 1, size=2)

        # roll the dice
        self.roll = double_dice(1, 6)

        # roll the dice until they are different
        while self.roll[0] == self.roll[1]:
            self.roll = double_dice(1, 6)

        # set the current agent
        if self.roll[0] > self.roll[1]:
            self.current_agent = WHITE
            self.roll = -self.roll
        else:
            self.current_agent = BLACK

        self.game = Game()
        self.counter = 0

        
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def render(self):
        if self.render_mode is None:
            return
        elif self.render_mode == 'human':
            self.game.render()
            return True
        else:
            if self.viewer is None:
                self.viewer = Viewer(SCREEN_W, SCREEN_H)

            if self.render_mode == 'rgb_array':
                width = SCREEN_W
                height = SCREEN_H

            else:
                assert self.render_mode == 'state_pixels', print(self.render_mode)
                width = STATE_W
                height = STATE_H

            return self.viewer.render(board=self.game.board, bar=self.game.bar, off=self.game.off, state_w=width, state_h=height)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_valid_actions(self, roll):
        return self.game.get_valid_plays(self.current_agent, roll)

    def get_opponent_agent(self):
        self.current_agent = self.game.get_opponent(self.current_agent)
        return self.current_agent


class BackgammonEnvPixel(BackgammonEnv):

    def __init__(self, render_mode='state_pixels'):
        super().__init__(render_mode)
        self.observation_space = Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)

    def step(self, action):
        observation, reward, done, winner = super().step(action)
        observation = self.render()
        return observation, reward, done, winner

    def reset(self):
        current_agent, roll, observation = super().reset()
        observation = self.render()
        return current_agent, roll, observation
