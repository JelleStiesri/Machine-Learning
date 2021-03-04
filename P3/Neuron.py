from math import e

class Neuron:

    def __init__(self, name: str, bias: float, weights: [float]):
        self.name = name
        self.bias = bias
        self.weights = weights


    def activation(self, inputs: list):
        """Dit is de activatiefunctie, hierin word de output van de neuron berekend.
        Hiervoor maken we eerst een gewogen som van alle inputs en voegen hier de bias aan toe.
        Ik stop vervolgens de gewogen som in de Sigmoid functie om de echte output te berekenen"""
        weighted_sum = 0
        for index in range(len(inputs)):  # Loop door indexen van de input lijst heen
            weighted_sum += inputs[index] * self.weights[index]  # Voegt de input waarde*Gewicht toe aan som
        weighted_sum += self.bias

        return 1 / (1 + e ** -weighted_sum)

    def error(self, expectations: ([], bool)):
        """Berekent de totale error over alle trainingsvoorbeelden
        Formule: MSE = Σ | d – y |2 / n"""
        error = 0
        for input_list, expectation in expectations:
            error += (expectation - self.activation(input_list))**2

        print(f'\n'
              f'Error: {error / len(expectations)}')


    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'


