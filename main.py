from Input_perceptron import *
from Perceptron import *
from PerceptronLayer import *
from PerceptronNetwork import *

def printOutput(output):
    print('- Output: ', output, '\n')

def PerceptronTest():
    """Invert Test"""

    input_not = [True]
    p_not = Perceptron('NOT', 0.5, [-1])
    output_NOT = p_not.activation(input_not)

    print(p_not)
    print('- Output: ', output_NOT, '\n')


    """AND Test"""

    input_AND = [True, True]
    p_and = Perceptron('AND', -1.5, [1, 1])
    output_AND = p_and.activation(input_AND)

    print(p_and)
    print('- Output: ', output_AND, '\n')

    """OR Test"""
    input_OR = [True, False]
    p_or = Perceptron('OR', -0.5, [0.5, 0.5])
    output_OR = p_or.activation(input_OR)

    print(p_or)
    print('- Output: ', output_OR, '\n')


    """NOR Test"""
    input_NOT = [True, False]

    p_or = Perceptron('OR', -0.5, [0.5, 0.5])
    output_ORNOT = p_or.activation(input_NOT)

    print(p_or)
    print('- Output: ', output_ORNOT)

    p_not = Perceptron('NOT', 0.5, [-1])
    output_NOT = p_not.activation([output_ORNOT])

    print(p_not)
    print('- Output: ', output_NOT)


def LayerTest():
    inputs = [True]
    p1 = Perceptron('P1 [NOT]', 0.5, [-1])
    print(p1, '\n')

    p2 = Perceptron('P2 [NOT]', 0.5, [-1])
    print(p2, '\n')

    layer1 = PerceptronLayer('Laag-1', [p1])
    output_layer1 = layer1.activation(inputs)

    layer2 = PerceptronLayer('Laag-2', [p2])
    output_layer2 = layer2.activation(output_layer1)

    printOutput(output_layer2)

def NetworkTest_XOR():
    p1 = Perceptron('P1 [AND]', -1, [0.5, 0.5])
    p2 = Perceptron('P2 [OR]', -1, [1, 1])
    p3 = Perceptron('P2 [XOR]', -1, [-1, 1])

    layer1 = PerceptronLayer('Laag-1 [Hidden]', [p1, p2])
    layer2 = PerceptronLayer('Laag-2 [Output]', [p3])

    network = PerceptronNetwork([layer1, layer2])

    input_list = [True, False]
    output = network.feed_forward(input_list)
    print(output)

def NetworkTest_ADDER():

    # Laag 1
    p1 = Perceptron('P2', -1, [1, 0])
    p2 = Perceptron('P1 [AND]', -1, [0.5, 0.5])
    p3 = Perceptron('P3', -1, [0, 1])
    layer1 = PerceptronLayer('Laag-1 [Hidden]', [p1, p2, p3])

    # Laag 2
    p4 = Perceptron('P4', -1, [1, 0, 0])
    p5 = Perceptron('P5 [NOT]', 0.5, [0, -1, 0])
    p6 = Perceptron('P6', -1, [0, 0, 1])
    layer2 = PerceptronLayer('Laag-2 [Hidden]', [p4, p5, p6])

    # Laag 3
    p7 = Perceptron('P7 [AND]', -1, [0.5, 0.5, 0])
    p8 = Perceptron('P8 [AND]', -1, [0, 0.5, 0.5])
    p9 = Perceptron('P9', -1, [0, 1, 0])
    layer3 = PerceptronLayer('Laag-2 [Hidden]', [p7, p8, p9])

    # Laag 4
    p10 = Perceptron('P10 [NOT]', 0.5, [-1, 0, 0])
    p11 = Perceptron('P11 [NOT]', 0.5, [0, -1, 0])
    p12 = Perceptron('P12', -1, [0, 0, 1])
    layer4 = PerceptronLayer('Laag-2 [Hidden]', [p10, p11, p12])

    # Laag 5
    p13 = Perceptron('P13 [AND]', -1, [0.5, 0.5, 0])
    p14 = Perceptron('P14 [AND??]', -1, [0, 0, 1])
    layer5 = PerceptronLayer('Laag-2 [Hidden]', [p13, p14])

    # Laag 6
    p15 = Perceptron('P15 [NOT]', 0.5, [-1, 0])
    p16 = Perceptron('P16 [NOT]', 0.5, [0, -1])
    layer6 = PerceptronLayer('Laag-2 [Hidden]', [p15, p16])


    network = PerceptronNetwork([layer1, layer2, layer3, layer4, layer5, layer6])

    input_list = [True, True]

    output = network.feed_forward(input_list)
    print(output)
    print(f'Carry: {int(output[1])} - Sum: {int(output[0])}')


# PerceptronTest()
# LayerTest()
# NetworkTest_XOR()
NetworkTest_ADDER()
