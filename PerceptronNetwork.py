from PerceptronLayer import *


class PerceptronNetwork:

    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, input_list):

        for index in range(len(self.layers)):
            layer = self.layers[index]
            input_list = layer.activation(input_list)
        output = input_list
        return output

