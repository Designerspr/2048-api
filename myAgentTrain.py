'''Build the test version of the agent.
'''
from game2048 import agents, game
import os
import numpy as np
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Input, Concatenate, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import to_categorical


class my_agent(agents.Agent):
    def __init__(self, game, max_depth=17, display=None):
        self.game = game
        self.display = display
        self.max_depth = max_depth
        self.model = self.build_model(max_depth=max_depth)

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        board_now = self.game.board
        board_input = np.expand_dims(
            board2input(board_now, self.max_depth), axis=0)
        # print(board_input.shape)
        choice = self.model.predict(board_input)[0]
        direction = np.where(np.max(choice) == choice)[0][0]
        # print(choice, direction)
        return direction

    def build_model(self, max_depth):
        '''Build the model of the class.
        '''
        x = Input(shape=(self.game.size, self.game.size, max_depth))
        # Conv Blocks
        y = self.add_blocks(x, 128)
        # y=self.add_blocks(y,256)
        # y=self.add_blocks(y,512)

        # Flatten&Dense Blocks
        y = AveragePooling2D(pool_size=(self.game.size, self.game.size))(y)
        y = Flatten()(y)
        for num in [1024, 1024]:
            y = Dense(num, kernel_initializer='he_uniform')(y)
            y = BatchNormalization()(y)
            y = LeakyReLU(alpha=0.2)(y)
        # Output
        y = Dense(4, activation='softmax')(y)
        model = Model(x, y)
        model.summary()
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model

    def add_blocks(self, inputs, num_filters):
        conv14 = Conv2D(
            num_filters,
            kernel_size=(1, 4),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv41 = Conv2D(
            num_filters,
            kernel_size=(4, 1),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv22 = Conv2D(
            num_filters,
            kernel_size=(2, 2),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv33 = Conv2D(
            num_filters,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        conv44 = Conv2D(
            num_filters,
            kernel_size=(4, 4),
            padding='same',
            kernel_initializer='he_uniform')(inputs)
        outputs = Concatenate(axis=-1)(
            [conv14, conv41, conv22, conv33, conv44])
        outputs = BatchNormalization(axis=-1)(outputs)
        outputs = LeakyReLU(alpha=0.2)(outputs)
        return outputs

    def train(self, imitatedAgent, batch_size=32, epoch=100, checkpoint=10):
        '''update myAgent.
        
        Arguments:
            imitatedAgent {Agent} -- the expert giving the righta answer.
        '''
        training_generator = training_gene(self, batch_size=batch_size)
        game_for_expert = game.Game(size=4, enable_rewrite_board=True)
        expert = imitatedAgent(game_for_expert, )
        for this_epoch in range(epoch):
            # readin next batch
            training_x = training_generator.__next__()
            training_y = list()
            # get the expert answer
            for i, each in enumerate(training_x):
                expert.game.board=each
                training_y.append(expert.step())
                training_x[i] = board2input(
                    training_x[i], max_depth=self.max_depth)
            # normolize the data
            training_x = np.array(training_x)
            training_y = to_categorical(training_y, num_classes=4)
            # train on batch
            loss,accu = self.model.train_on_batch(training_x, training_y)
            # print module
            print('epoch:%d/%d;loss=%.2f ;accu=%.2f' % (this_epoch, epoch, loss,accu))
            if np.mod(this_epoch, checkpoint) == 0:
                # check point : show how the net works with a real game.
                empty_board = np.zeros((self.game.size, self.game.size))
                self.game.board = empty_board
                self.game._maybe_new_entry()
                self.game._maybe_new_entry()
                self.game.__end = False

                n_iter = 0
                while not self.game.end:
                    direction = self.step()
                    self.game.move(direction)
                    n_iter += 1

        return

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def save_model(self, path):
        keras.models.save_model(self.model, path)


def board2input(board, max_depth):
    '''reshape the board into the one that the network used.
    
    Arguments:
        board {ndarray}
        max_depth {int} -- the depth.
    '''
    size = len(board)
    board_input = np.zeros((size, size, max_depth))
    meshx, meshy = np.meshgrid(range(size), range(size))
    meshx, meshy = meshx.flatten(), meshy.flatten()
    for (x, y) in zip(meshx, meshy):
        if board[x, y] != 0:
            pos = int(np.log2(board[x, y]) - 1)
            board_input[x, y, pos] = 1
    # print(board_input.shape)
    return board_input


def training_gene(weak_agent, batch_size=32):
    '''generate training board data.
    using random isn't useful due to so many conditions to be solved.
    
    this is the primary version. one for all.

    Arguments:
        weak_agent {Agent.agent} -- the agent to be trained.
    '''
    buffer = list()

    while True:
        # init new game and play
        empty_board = np.zeros((weak_agent.game.size, weak_agent.game.size))
        weak_agent.game.board = empty_board
        weak_agent.game._maybe_new_entry()
        weak_agent.game._maybe_new_entry()
        weak_agent.game.__end = False

        # Run and record the board only
        while not weak_agent.game.end:
            buffer.append(weak_agent.game.board)
            direction = weak_agent.step()
            weak_agent.game.move(direction)

        # output the buffer
        while len(buffer) > batch_size:
            batch = buffer[:batch_size]
            yield (batch)
            buffer = buffer[batch_size:]


new_game = game.Game(size=4, enable_rewrite_board=True)
ag = my_agent(new_game)
ag.train(agents.ExpectiMaxAgent, batch_size=32, epoch=1000, checkpoint=10)
ag.save_model('1.h5')
