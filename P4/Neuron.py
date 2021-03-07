from math import e
from typing import List

class Neuron:

    def __init__(self, name: str, bias: float, weights: List[float]):
        self.error = None
        self.name = name
        self.bias = bias
        self.weights = weights

    def sigmoid(self, weighted_sum: float):
        return 1 / (1 + e ** -weighted_sum)

    def activation(self, inputs: list):
        """Dit is de activatiefunctie, hierin word de output van de neuron berekend.
        Hiervoor maken we eerst een gewogen som van alle inputs en voegen hier de bias aan toe.
        Ik stop vervolgens de gewogen som in de Sigmoid functie om de echte output te berekenen"""
        weighted_sum = self.bias
        for index in range(len(inputs)):  # Loop door indexen van de input lijst heen
            weighted_sum += inputs[index] * self.weights[index]  # Voegt de input waarde*Gewicht toe aan som

        return self.sigmoid(weighted_sum)

    # def calculate_error(self, expectation: [[], bool]):
    #     for input_list, target in [expectation]:
    #         derivative = self.activation(input_list) * (1 - self.activation(input_list))
    #         # print(f'Afgeleide {derivative}\n')
    #         error = derivative * -(target - self.activation(input_list))
    #         # print(f'Error: {error}')
    #
    #         self.error = error

    def calculate_error(self, output: float,  target: float):
        derivative = output * (1 - output)
        error = derivative * -(target - output)

        self.error = error

    def update(self, learning_rate: float = 0.1):
        self.bias -= (learning_rate*self.error)

        pass

    # def update(self, expectations: [[], bool]):
    #     for input_list, expectation in expectations:
    #         output = self.activation(input_list)
    #         print(f'input: {input_list, expectation} --- Output: {output}')
    #
    #         self.calculate_error(output, expectation)
    #         print(f'Error: {self.error}')
    #
    #         gradient = output * self.error
    #         print(f'Gradient: {gradient}')
    #
    #
    #         print("\n")

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'


