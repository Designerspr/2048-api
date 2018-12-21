'''Build the test version of the agent.
'''
from game2048 import agents,game
import os
import numpy as np
import keras
from  keras.layers import Dense, Conv2D, BatchNormalization,Flatten,Input,Concatenate,AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU

class my_agent(agents.Agent):

    def __init__(self, game,max_depth=20,display=None):
        self.game = game
        self.display = display
        self.model=self.build_model(max_depth=max_depth)

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
        return
    
    def build_model(self,max_depth=15):
        '''Build the model of the class.
        '''
        x=Input(shape=(self.game.size,self.game.size,max_depth))
        # Conv Blocks
        y=self.add_blocks(x,128)
        # y=self.add_blocks(y,256)
        # y=self.add_blocks(y,512)
        
        # Flatten&Dense Blocks 
        y=AveragePooling2D(pool_size=(self.game.size,self.game.size))(y)
        y=Flatten()(y)
        for num in [1024,1024]:
            y=Dense(num,kernel_initializer='he_uniform')(y)
            y=BatchNormalization()(y)
            y=LeakyReLU(alpha=0.2)(y)
        # Output
        y=Dense(4,activation='softmax')(y)
        model=Model(x,y)
        model.summary()
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


    def add_blocks(self,inputs,num_filters):
        conv14=Conv2D(num_filters,kernel_size=(1,4),kernel_initializer='he_uniform')(inputs)
        conv41=Conv2D(num_filters,kernel_size=(4,1),kernel_initializer='he_uniform')(inputs)
        conv22=Conv2D(num_filters,kernel_size=(2,2),kernel_initializer='he_uniform')(inputs)
        conv33=Conv2D(num_filters,kernel_size=(3,3),kernel_initializer='he_uniform')(inputs)
        conv44=Conv2D(num_filters,kernel_size=(4,4),kernel_initializer='he_uniform')(inputs)
        outputs=Concatenate([conv14,conv41,conv22,conv33,conv44])
        outputs=BatchNormalization(axis=-1)(outputs)
        outputs=LeakyReLU(alpha=0.2)(outputs)
        return outputs

    def train(imitatedAgent):
        '''Update
        
        Arguments:
            imitatedAgent {[type]} -- [description]
        '''
        return

ag=my_agent(game.Game(4),20)