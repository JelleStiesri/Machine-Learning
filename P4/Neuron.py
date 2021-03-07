from math import e
from typing import List


class Neuron:

    def __init__(self, name: str, neuron_type: str, bias: float, weights: List[float]):
        self.error = None
        self.name = name
        self.neuron_type = neuron_type
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

    def calculate_error(self, output: float, target: float):
        derivative = output * (1 - output)  # σ'(inputj) = σ(inputj) ∙ (1 – σ(inputj)) = outputj ∙ (1 – outputj)
        error = derivative * -(target - output)  # Δj = σ'(inputj) ∙ –(targetj – outputj)

        self.error = error

    def update(self, output: float, learning_rate: float = 0.1):
        self.bias -= (learning_rate * self.error)  # Δbj = η ∙ Δj
        for i in range(len(self.weights)):
            self.weights[i] -= (learning_rate * output * self.error)  # Δwi,j = η ∙ ∂C/∂wi,j = η ∙ outputi ∙ Δj

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'
