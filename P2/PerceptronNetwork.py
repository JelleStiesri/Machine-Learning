class PerceptronNetwork:

    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, input_list):
        """Loopt door de layers heen en geeft de output van de vorige
        als input voor de volgende."""
        for index in range(len(self.layers)):
            layer = self.layers[index]  # selecteerde de layer
            input_list = layer.activate(input_list)
        return input_list

    def __str__(self):
        return f'----------------------------------------\n' \
               f'Dit netwerk bestaat uit {len(self.layers)} layers\n'
