from math import e
from typing import List


class Neuron:

    def __init__(self, name: str, bias: float, weights: List[float]):
        self.error = None
        self.output = None
        self.name = name
        self.bias = bias
        self.weights = weights


    def calculate_weighted_sum(self, weight_list: List[float], input_list: List):
        weighted_sum = 0
        for index in range(len(input_list)):  # Loop door indexen van de input lijst heen
            weighted_sum += input_list[index] * weight_list[index]  # Voegt de input waarde*Gewicht toe aan som

        return weighted_sum

    def sigmoid(self, weighted_sum: float):
        return 1 / (1 + e ** -weighted_sum)

    def activation(self, inputs: List):
        """Dit is de activatiefunctie, hierin word de output van de neuron berekend.
        Hiervoor maken we eerst een gewogen som van alle inputs en voegen hier de bias aan toe.
        Ik stop vervolgens de gewogen som in de Sigmoid functie om de echte output te berekenen"""
        weighted_sum = self.calculate_weighted_sum(self.weights, inputs) + self.bias

        output = self.sigmoid(weighted_sum)
        self.output = output
        return output

    def calculate_derivative(self, output: float):
        return output * (1 - output)  # σ'(inputj) = σ(inputj) ∙ (1 – σ(inputj)) = outputj ∙ (1 – outputj)

    def calculate_error_output(self, output: float, target: float):
        # print("\n==============================Output Neuron Error==============================")
        # print(self)
        # print(f'output: {output} - Target {target}')
        # print(f'derivative: {self.calculate_derivative(output)}')
        # print(f'floep: {target - output}')
        error = self.calculate_derivative(output) * -(target - output)  # Δj = σ'(inputj) ∙ –(targetj – outputj)



        self.error = error
        # print(error)
        return error

    def calculate_error_hidden(self, output: float, prev_weights: List, prev_errors: List):
        # print("\n----------------------Hidden Neuron Error-----------------------------------")
        # print(self)
        # print(f'Prev Erro: {prev_errors}')
        # print(f'Prev Weight: {prev_weights}')
        # print(self.calculate_weighted_sum(prev_weights, prev_errors))
        error = self.calculate_derivative(output) * self.calculate_weighted_sum(prev_weights, prev_errors)

        self.error = error
        # print(error)
        return error

    def update(self, prev_outputs: List, learning_rate: float = 10):
        self.bias -= (learning_rate * self.error)  # Δbj = η ∙ Δj
        for i in range(len(self.weights)):
            # print((learning_rate * prev_outputs[i] * self.error))
            self.weights[i] -= (learning_rate * prev_outputs[i] * self.error)  # Δwi,j = η ∙ ∂C/∂wi,j = η ∙ outputi ∙ Δj

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'


