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

N1 = Neuron('Neuron 1', -1.2, [0.6, 0.2])
N2 = Neuron('Neuron 2', -1.2, [0.6, 0.2])
N3 = Neuron('Neuron 3', -1.2, [0.6, 0.2])

outputNeuron1 = Neuron('Output Neuron', -1.2, [0.5, 0.1, 1])
outputNeuron2 = Neuron('Output Neuron', -1.2, [0.5, 0.15, 0.8])

hiddenLayer = NeuronLayer('Hidden Layer', [N1, N2, N3])
# outputLayer = NeuronLayer('Output Layer', [outputNeuron])
outputLayer = NeuronLayer('Output Layer', [outputNeuron1, outputNeuron2])

network_xor = NeuronNetwork([hiddenLayer, outputLayer])
print(network_xor)

adder_expectation = [[[False, False], [False, False]],
                   [[False, True], [True, False]],
                   [[True, False], [True, False]],
                   [[True, True], [False, True]]]

print_table(adder_expectation, "Verwachting")

old_output = create_table_data(network_xor.feed_forward, adder_expectation)
print_table(old_output, 'Uitkomst VOOR training')

network_xor.train(adder_expectation, 25000)
adder_expectation = [[[False, False], [False, False]],
                   [[False, True], [True, False]],
                   [[True, False], [True, False]],
                   [[True, True], [False, True]]]



new_output = create_table_data(network_xor.feed_forward, adder_expectation)
print_table(new_output, 'Uitkomst NA training')

print(outputNeuron1, outputNeuron1.output)
print('\n')
print(outputNeuron2, outputNeuron2.output)
# inputs = [[False, False],
#           [False, True],
#           [True, False],
#           [True, True]]
# targets = [False, True, True, False]

# inputs = [[False, True]]
# targets = [True, False]
# print_table(xor_expectation, "Verwachting")


# old_output = create_table_data(network_xor.feed_forward, xor_expectation)
# print_table(old_output, 'Uitkomst voor training')
# print(N1.output, N2.output, outputNeuron.output)

# network_xor.feed_forward()
# network_xor.train(inputs, targets)