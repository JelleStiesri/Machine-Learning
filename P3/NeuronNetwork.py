class NeuronNetwork:

    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, input_list):
        """Loopt door de layers heen en geeft de output van de vorige
        als input voor de volgende."""
        for index in range(len(self.layers)):
            layer = self.layers[index]  # selecteerde de layer
            input_list = layer.activate(input_list)
        return input_list

    def error(self, expectations: ([], bool)):
        """Berekent de totale error over alle trainingsvoorbeelden
        Formule: MSE = Σ | d – y |2 / n"""
        error = 0
        for input_list, expectation in expectations:
            outcome = self.feed_forward(input_list)
            for item in range(len(outcome)):
                # error += (expectation[item] - self.activation(input_list))**2
                # error += (expectation - self.feed_forward(input_list))**2
                error += (expectation[item] - outcome[item])**2

        print(f'\n'
              f'Error: {error / len(expectations)}')

    def __str__(self):
        return f'----------------------------------------\n' \
               f'Dit netwerk bestaat uit {len(self.layers)} layers\n'
