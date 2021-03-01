class NeuronLayer:

    def __init__(self, name, neurons):
        self.name = name
        self.neurons = neurons

    def activate(self, inputs):
        output = []

        for neuron in self.neurons:
            """Voer de activatiefunctie uit in elke neurons in de laag"""
            if len(self.neurons) == 1:
                return neuron.activation(inputs)
            output.append(neuron.activation(inputs))  # append de output van de neurons

        return output

    def __str__(self):
        return f'----------------------------------------\n'\
               f'<{self.name}> ' \
               f'- Deze layer bestaat uit {len(self.neurons)} neurons\n'
    