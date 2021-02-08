class PerceptronLayer:

    def __init__(self,name, perceptrons):
        self.name = name
        self.perceptrons = perceptrons

    def activation(self, inputs):
        output = []

        for perceptron in self.perceptrons:
            output.append(perceptron.activation(inputs))

        return output

    def __str__(self):
        return f'Deze layer bestaat uit {len(self.perceptrons)} Perceptrons'