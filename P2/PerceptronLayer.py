class PerceptronLayer:

    def __init__(self,name, perceptrons):
        self.name = name
        self.perceptrons = perceptrons

    def activate(self, inputs):
        output = []

        for perceptron in self.perceptrons:
            """Voer de activatiefunctie uit in elke perceptron in de laag"""
            if len(self.perceptrons) == 1:
                return perceptron.activation(inputs)
            output.append(perceptron.activation(inputs))  # append de output van de perceptron

        return output

    def __str__(self):
        return f'----------------------------------------\n'\
               f'<{self.name}> ' \
               f'- Deze layer bestaat uit {len(self.perceptrons)} Perceptrons\n'
    