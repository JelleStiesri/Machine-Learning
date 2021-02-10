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

        """Hieronder maak ik een string waarin de berekening van de gewogen som laat zien. Kon geen betere plek
        vinden omdat de __str__ functie geen parameters kan hebben en ik de inputs niet als atribuut mag opslaan"""
        string = '\n'
        string += f'Input: {inputs} \n'
        for item in range(len(inputs)):
            string += f'{int(inputs[item])}*{self.weights[item]}' # Voegt elke input + weight toe aan string
            if item == len(inputs)-1:
                string += f' + {self.bias}(bias) = '
            else:
                string += ' + '
        string += f'{weighted_sum}\n'
        print(string)

        return weighted_sum + self.bias >= 0  # Returnt 1 = True als het getal boven 0 komt.

    def __str__(self):
        return f'<{self.name} port>\n' \
               f'- Bias = {self.bias}\n' \
               f'- Weights = {self.weights}'

