from P4.Neuron import *
from P4.NeuronLayer import *
from P4.NeuronNetwork import *
import random
import pandas as pd


def create_table_data(function, expectations):
    """Deze functie berekent de uitkomst van een perceptron/netwerk en geeft dit terug in een goed format"""
    data = []

    for item in expectations:
        output = function(item[0])
        data.append([item[0], output])
    return data


def print_table(data, caption):
    """Deze functie print een waarheidstabel"""

    columns = ['Input', 'Output']

    df = pd.DataFrame.from_records(data, columns=columns)
    print(f'<{caption}>\n {df}\n')

N1 = Neuron('Neuron 1', -1.3, [0.6, 0.2])
N2 = Neuron('Neuron 2', -1.3, [0.6, 0.2])
outputNeuron = Neuron('Output Neuron', -1.2, [0.5, 0.1])
# outputNeuron2 = Neuron('Output Neuron', -1.2, [0.5, 0.1])

hiddenLayer = NeuronLayer('Hidden Layer', [N1, N2])
outputLayer = NeuronLayer('Output Layer', [outputNeuron])
# outputLayer = NeuronLayer('Output Layer', [outputNeuron, outputNeuron2])

network_xor = NeuronNetwork([hiddenLayer, outputLayer])
print(network_xor)

xor_expectation = [[[False, False], False],
                   [[False, True], True],
                   [[True, False], True],
                   [[True, True], False]]

inputs = [[False, False],
          [False, True],
          [True, False],
          [True, True]]
targets = [False, True, True, False]

# inputs = [[False, False]]
# targets = [False]
print_table(xor_expectation, "Verwachting")


old_output = create_table_data(network_xor.feed_forward, xor_expectation)
print_table(old_output, 'Uitkomst VOOR training')
# print(N1.output, N2.output, outputNeuron.output)

# network_xor.feed_forward()

# print(f'\nVOOR UPDATE\n{N1}\n\n {N2}\n\n {outputNeuron}')
network_xor.train(xor_expectation, 25)

xor_expectation = [[[False, False], False],
                   [[False, True], True],
                   [[True, False], True],
                   [[True, True], False]]
new_output = create_table_data(network_xor.feed_forward, xor_expectation)
print_table(new_output, 'Uitkomst NA training')


# print(f'\nNA UPDATE\n{N1}\n\n {N2}\n\n {outputNeuron}')