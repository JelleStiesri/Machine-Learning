from Input_perceptron import *
from Perceptron import *

"""Invert Test"""

x1 = Input_perceptron(True)
p1 = Perceptron('NOT', 0.5, [-1], [x1])
p1.activation()

print(p1, '\n')


"""Invert Test"""
x1 = Input_perceptron(True)
x2 = Input_perceptron(True)
p1 = Perceptron('AND', -1.5, [1, 1], [x1, x2])
p1.activation()

print(p1)