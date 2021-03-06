class Perceptron:

    def __init__(self, name, bias, weights):
        self.name = name  # Naam heeft geen functionele waarde, is alleen voor overzicht
        self.bias = bias
        self.weights = weights

    def activation(self, inputs):
        """Dit is de activatiefunctie, hierin word de output van de perceptron berekend.
        Hiervoor maken we eerst een gewogen som van alle inputs en voegen hier de bias aan toe.
        Wanneer dit getal boven de 0 komt word er 1 gereturnt."""

        weighted_sum = 0
        for index in range(len(inputs)):  # Loop door indexen van de input lijst heen
            weighted_sum += inputs[index] * self.weights[index]  # Voegt de input waarde*Gewicht toe aan som

        return weighted_sum + self.bias >= 0  # Returnt 1 = True als het getal boven 0 komt.

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}\n'
    