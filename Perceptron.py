from Input_perceptron import *

class Perceptron:

    def __init__(self, name, bias, weights, input_layer):
        self.name = name
        self.bias = bias
        self.weights = weights
        self.input_layer = input_layer
        self.value = None  # Aan het begin heeft een perceptron nog geen waarde

    def activation(self):
        """Dit is de activatiefunctie, hierin word de output van de perceptron berekend.
        Hiervoor maken we eerst een gewogen som van alle inputs en voegen hier de bias aan toe.
        Wanneer dit getal boven de 0 komt word er 1 gereturnt."""
        weighted_sum = 0
        for index in range(len(self.input_layer)):  # Loop door indexen van de input lijst heen
            weighted_sum += self.input_layer[index].get_value() * self.weights[index]  # Voegt de input waarde*Gewicht toe aan som

        if weighted_sum + self.bias > 0:  # Returnt 1 = True als het getal boven 0 komt.
            self.value = True
        else:
            self.value = False

    def __str__(self):
        return f'Dit is een {self.name} Poort ' \
               f'- Bias = {self.bias} \n' \
               f'- Waarde = {self.value}'

