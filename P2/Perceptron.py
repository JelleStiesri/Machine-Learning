from typing import List

class Perceptron:

    def __init__(self, name: str, bias: float, weights: List[float]):
        self.name = name  # Naam heeft geen functionele waarde, is alleen voor overzicht
        self.bias = bias
        self.weights = weights

    def step_function(self, weighted_sum: float):
        return weighted_sum + self.bias >= 0  # Returnt 1 = True als het getal boven 0 komt.

    def activation(self, inputs: list):
        """Dit is de activatiefunctie, hierin word de output van de perceptron berekend.
        Hiervoor maken we eerst een gewogen som van alle inputs en voegen hier de bias aan toe.
        Wanneer dit getal boven de 0 komt word er 1 gereturnt."""

        weighted_sum = 0
        for index in range(len(inputs)):  # Loop door indexen van de input lijst heen
            weighted_sum += inputs[index] * self.weights[index]  # Voegt de input waarde*Gewicht toe aan som

        return self.step_function(weighted_sum)

    def update(self, expectations: ([], bool), loops: int, learning_rate: float = 0.1):
        """De update functie voert de learning rule uit over het algoritme
        Dit pas de Weights en bias aan."""
        for loop in range(loops):
            for input_list, expectation in expectations:
                output = self.activation(input_list)  # (y = f(w ∙ x))

                if output != expectation:
                    """Deze if statement is niet nodig, het zorgt er alleen voor dat er geen 
                    regels code worden gerund die niet nodig zijn. Zo word de code efficienter"""

                    error = float(expectation - output)  # (e = d – y)
                    self.bias += (learning_rate * error)  # (Δb = η ∙ e)  en (b' = b + Δb)

                    for index in range(len(self.weights)):
                        self.weights[index] += (learning_rate * error * input_list[index])  # (Δw = η ∙ e ∙ x) en (w'
                        # = w + Δw)

    def error(self, expectations: ([], bool)):
        """Berekent de totale error over alle trainingsvoorbeelden
        Formule: MSE = Σ | d – y |2 / n"""
        error = 0
        for input_list, expectation in expectations:
            error += (expectation - self.activation(input_list))**2

        return error / len(expectations)

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'
