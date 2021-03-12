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
        """Deze functie berekend een gewogen som van een lijst gewichte en een lijst inputs"""
        weighted_sum = 0
        for index in range(len(input_list)):  # Loop door indexen van de input lijst heen
            weighted_sum += input_list[index] * weight_list[index]  # Voegt de input waarde*Gewicht toe aan som

        return weighted_sum

    def sigmoid(self, weighted_sum: float):
        """Deze functie voert de Sigmoid activatiefunctie uit"""
        return 1 / (1 + e ** -weighted_sum)

    def activation(self, inputs: List):
        """Hier word de gewogen som van de inputs/gewichten berekend en vervolgens de output van de neuron """
        weighted_sum = self.calculate_weighted_sum(self.weights, inputs) + self.bias

        output = self.sigmoid(weighted_sum)
        self.output = output
        return output

    def calculate_derivative(self, output: float):
        """Deze functie berekend de afgeleide van de sigmoid functie"""
        return output * (1 - output)  # σ'(inputj) = σ(inputj) ∙ (1 – σ(inputj)) = outputj ∙ (1 – outputj)

    def calculate_error_output(self, output: float, target: float):
        """Deze functie berekend de error van een outputNeuron"""
        error = self.calculate_derivative(output) * -(target - output)  # Δj = σ'(inputj) ∙ –(targetj – outputj)

        self.error = error

        return error

    def calculate_error_hidden(self, output: float, prev_weights: List, prev_errors: List):
        """Deze functie berekend de error van een hiddenNeuron"""
        error = self.calculate_derivative(output) * self.calculate_weighted_sum(prev_weights, prev_errors)  # Δi = σ'(inputi) ∙ Σj wi,j ∙ Δj

        self.error = error

        return error

    def update(self, prev_outputs: List, learning_rate: float):
        """Deze functie berekend de delta's voor gewicht&bias en voert deze door
        Error bereken moet van tevoren, omdat de gewichten van andere Neuronen daar nog invloed op kunnen hebben
        De delta's & update tegelijk maakt geen verschil"""
        self.bias -= (learning_rate * self.error)  # Δbj = η ∙ Δj
        for i in range(len(self.weights)):  # Voor elk gewicht..
            self.weights[i] -= (learning_rate * prev_outputs[i] * self.error)  # Δwi,j = η ∙ ∂C/∂wi,j = η ∙ outputi ∙ Δj

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'


