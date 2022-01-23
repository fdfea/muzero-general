import datetime
import os

import numpy as np

from enum import Enum
from games.abstract_game import AbstractGame

class Rule(Enum):
    XA   = 0x8001FFFFFFFFFFFF
    XR1  = 0x8001FFFFFEFFFFFF
    XR1U = 0x80000001C2870000
    XR1I = 0x80000001C3870000
    XR2  = 0x8001FFFE3C78FFFF
    XR2U = 0x800001F224489F00
    XR2I = 0x800001F3E7CF9F00
    XR2C = 0x8001FFFE3D78FFFF
    XR3  = 0x8001FE0C183060FF
    XR3C = 0x8001FE0C193060FF
    OA   = 0x0001FFFFFFFFFFFF
    OR1  = 0x0001FFFFFEFFFFFF
    OR1U = 0x00000001C2870000
    OR1I = 0x00000001C3870000
    OR2  = 0x0001FFFE3C78FFFF
    OR2U = 0x000001F224489F00
    OR2I = 0x000001F3E7CF9F00
    OR2C = 0x0001FFFE3D78FFFF
    OR3  = 0x0001FE0C183060FF
    OR3C = 0x0001FE0C193060FF

class Rules(Enum):
    def __getitem__(self, index):
        return self._value_[index]
    CLASSICAL = [
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
    ]
    MODERN = [
        Rule.XA.value, Rule.OR1I.value, Rule.OR3.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
    ]
    EXPERIMENTAL = [
        Rule.XA.value, Rule.OR2.value, Rule.OR3.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
        Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value,
        Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value, Rule.XA.value, Rule.OA.value,
    ]

class TicTacTrains:
    _ROWS = 7
    _COLUMNS = 7
    _LENGTH = _ROWS * _COLUMNS
    _LEFT = ord('a')
    _BOTTOM = ord('0')

    def __init__(self, rules):
        self._move = 0
        self._data = np.zeros(self._LENGTH, dtype=bool)
        self._valid = np.zeros(self._LENGTH, dtype=bool)
        self._rules = rules
    def copy(self):
        game = TicTacTrains()
        game._move = self._move
        game._data = self._data
        game._valid = self._valid
        game._rules = self._rules
        return game

    def player(self):
        return self._rules[self._move] & 0x8000000000000000 != 0
    def finished(self):
        return self._move >= self._LENGTH
    def valid(self):
        return ~self._valid & self.array(self._rules[self._move])
    def move(self, index):
        if self.player():
            self.set(self._data, index)
        self.set(self._valid, index)
        self._move += 1
    def indices(self, adjacent=False):
        valid = self.valid()
        return {index for index in range(self._LENGTH)
            if self.empty(index) and self.test(valid, index) and (not adjacent or self.adjacent(index))}
    def score(self):
        x = o = 1
        for index in range(self._LENGTH):
            if not self.empty(index):
                score = self.path(index, np.zeros(self._LENGTH, dtype=bool))
                if self.iplayer(index):
                    x = max(x, score)
                else:
                    o = max(o, score)
        return x - o

    def iplayer(self, index):
        return self.test(self._data, index)
    def empty(self, index):
        return not self.test(self._valid, index)
    def leftv(self, index):
        return index % self._COLUMNS > 0
    def left(self, index):
        return index - 1
    def rightv(self, index):
        return index % self._COLUMNS < (self._COLUMNS - 1)
    def right(self, index):
        return index + 1
    def topv(self, index):
        return index >= self._COLUMNS
    def top(self, index):
        return index - self._COLUMNS
    def bottomv(self, index):
        return index < self._ROWS * (self._COLUMNS - 1)
    def bottom(self, index):
        return index + self._COLUMNS
    def adjacent(self, index):
        return self.leftv(index) and not self.empty(self.left(index)) \
        or self.rightv(index) and not self.empty(self.right(index)) \
        or self.topv(index) and not self.empty(self.top(index)) \
        or self.bottomv(index) and not self.empty(self.bottom(index)) \
        or self.leftv(index) and self.topv(index) and not self.empty(self.left(self.top(index))) \
        or self.leftv(index) and self.bottomv(index) and not self.empty(self.left(self.bottom(index))) \
        or self.rightv(index) and self.topv(index) and not self.empty(self.right(self.top(index))) \
        or self.rightv(index) and self.bottomv(index) and not self.empty(self.right(self.bottom(index)))
    def path(self, index, visited):
        length = 0
        player = self.iplayer(index)
        self.set(visited, index)
        if self.leftv(index) \
            and not self.empty(self.left(index)) \
            and self.iplayer(self.left(index)) is player \
            and not self.test(visited, self.left(index)):
            length = max(length, self.path(self.left(index), visited))
        if self.rightv(index) \
            and not self.empty(self.right(index)) \
            and self.iplayer(self.right(index)) is player \
            and not self.test(visited, self.right(index)):
            length = max(length, self.path(self.right(index), visited))
        if self.topv(index) \
            and not self.empty(self.top(index)) \
            and self.iplayer(self.top(index)) is player \
            and not self.test(visited, self.top(index)):
            length = max(length, self.path(self.top(index), visited))
        if self.bottomv(index) \
            and not self.empty(self.bottom(index)) \
            and self.iplayer(self.bottom(index)) is player \
            and not self.test(visited, self.bottom(index)):
            length = max(length, self.path(self.bottom(index), visited))
        self.reset(visited, index)
        return length + 1
    
    def test(self, array, index):
        return array[index] == True
    def set(self, array, index):
        array[index] = True
    def reset(self, array, index):
        array[index] = False
    def array(self, num):
        return np.array(list(np.binary_repr(num).zfill(self._LENGTH)[-self._LENGTH:])).astype(bool)

    def char(self, index):
        return ' ' if self.empty(index) else 'X' if self.iplayer(index) else 'O'
    def string(self):
        bldr = list()
        for index in range(self._LENGTH):
            if index % self._ROWS == 0:
                bldr.append(f"{str(self._ROWS - index // self._ROWS)} ")
            bldr.append(f"[{self.char(index)}]")
            if (index + 1) % self._COLUMNS == 0:
                bldr.append('\n')
        bldr.append("& ")
        for index in range(self._COLUMNS):
            bldr.append(f" {chr(index + self._LEFT)} ")
        return str().join(bldr);
    def index(self, id):
        if len(id) != 2:
            return -1
        file = ord(id[0])
        rank = ord(id[1])
        if file < self._LEFT \
            or file >= (self._LEFT + self._COLUMNS) \
            or (rank - self._BOTTOM) <= 0 \
            or (rank - self._BOTTOM) > self._ROWS:
            return -1
        return self._COLUMNS * (self._ROWS - (rank - self._BOTTOM)) + (file - self._LEFT)
    def id(self, index):
        if index >= 0 and index < self._LENGTH:
            return str().join(list(chr((index % self._COLUMNS) + self._LEFT),
                chr((self._ROWS - (index / self._ROWS)) + self._BOTTOM)))
        return "?"

"""
def main():
    game = TicTacTrains(Rules.CLASSICAL)
    print(game.string(), end="\n\n")
    while not game.finished():
        indices = game.indices()
        while True:
            id = input("Enter move: ")
            index = game.index(id)
            if index in indices:
                game.move(index)
                print(game.string(), end="\n\n")
                break
    print(f"Score: {game.score()}")

if __name__ == "__main__":
    main()
"""

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0 # Seed for numpy, torch and the game
        self.max_num_gpus = 1 # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 7, 7) # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(TicTacTrains._LENGTH)) # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2)) # List of players. You should only edit the length
        self.stacked_observations = 0 # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0 # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1 # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 49 # Maximum number of moves if game is not finished before
        self.num_simulations = 49 # Number of future moves self-simulated
        self.discount = 1 # Chronological discount of the reward
        self.temperature_threshold = None # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.1
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet" # "resnet" / "fullyconnected"
        self.support_size = 1 # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1 # Number of blocks in the ResNet
        self.channels = 16 # Number of channels in the ResNet
        self.reduced_channels_reward = 16 # Number of channels in reward head
        self.reduced_channels_value = 16 # Number of channels in value head
        self.reduced_channels_policy = 16 # Number of channels in policy head
        self.resnet_fc_reward_layers = [8] # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8] # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8] # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [] # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16] # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16] # Define the hidden layers in the reward network
        self.fc_value_layers = [] # Define the hidden layers in the value network
        self.fc_policy_layers = [] # Define the hidden layers in the policy network

        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) # Path to store the model weights and TensorBoard logs
        self.save_model = True # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000 # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 100 # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10 # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25 # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = False # Train on GPU if available

        self.optimizer = "Adam" # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4 # L2 weights regularization
        self.momentum = 0.9 # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        # Replay Buffer
        self.replay_buffer_size = 3000 # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20 # Number of game moves to keep for every batch element
        self.td_steps = 20 # Number of steps in the future to take into account for calculating the target value
        self.PER = True # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5 # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0 # Number of seconds to wait after each played game
        self.training_delay = 0 # Number of seconds to wait after each training step
        self.ratio = None # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        return 1.

class Game(AbstractGame):
    def __init__(self, seed=None):
        self.game = TicTacTrains(rules=Rules.CLASSICAL)

    # reward here is probably wrong
    def step(self, action):
        self.game.move(action)
        finished = self.game.finished()
        reward = self.game.score() if finished else 0
        return self.state(), reward, finished

    def to_play(self):
        return int(self.game.player())

    def legal_actions(self):
        return np.array(list(self.game.indices(adjacent=True)))

    def reset(self):
        self.game = TicTacTrains(rules=self.game._rules)
        return self.state()

    def render(self):
        print(self.game.string(), end='\n\n')
        input("Press enter to take a step: ")

    def human_to_action(self):
        indices = self.game.indices()
        while True:
            id = input("Enter move: ")
            index = self.game.index(id)
            if index in indices:
                return index

    def expert_agent(self):
        pass

    def action_to_string(self, action):
        return self.game.id(action)

    def state(self):
        xstate = np.reshape(self.game._valid & self.game._data, (self.game._ROWS, self.game._COLUMNS))
        ostate = np.reshape(self.game._valid & ~self.game._data, (self.game._ROWS, self.game._COLUMNS))
        empty = np.reshape(~self.game._valid, (self.game._ROWS, self.game._COLUMNS))
        return np.array([xstate, ostate, empty])
