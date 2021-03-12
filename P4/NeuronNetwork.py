from Neuron import *
import random


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

    def train(self, dataSet, epochs, learning_rate):
        """In deze functie word een Netwerk van Neuronen getraind."""
        for _ in range(epochs):  # Aantal Epochs = aantal keer dat de hele dataset getraind word.
            random.shuffle(dataSet)  # De dataset word elke epoch geshuffled zodat het netwerk niet specifiek op een bepaalde volgorde word getraind.

            inputs = []
            targets = []
            for row in dataSet:
                inputs.append(row[0])
                targets.append(row[1])

            for target_index, input_list in enumerate(inputs):
                self.feed_forward(input_list)  # Voert het hele netwerk uit zodat de errors&delta's berekend kunnen worden
                target = targets[target_index]

                """Error berekenen voor de OUTPUT NEURONS"""
                for output_neuron_index, output_neuron in enumerate(self.layers[-1].neurons):
                    output_neuron.calculate_error_output(output_neuron.output, target[output_neuron_index])

                """Error berekenen voor de HIDDEN NEURONS"""
                reversed_layers = list(reversed(self.layers))  # Reversed omdat we van achter naar voren werken

                for layer_index, layer in enumerate(reversed_layers[1:]):  # Hierin nemen we niet de output layer mee
                    for hidden_neuron_index, hidden_neuron in enumerate(layer.neurons):
                        # De Error en weights van de vorige laag worden meegegeven aan de laag daarvoor om de error te kunnen berekenen
                        prev_errors = []
                        prev_weights = []
                        for neuron in reversed_layers[layer_index].neurons:
                            prev_weights.append(neuron.weights[hidden_neuron_index])
                            prev_errors.append(neuron.error)

                        hidden_neuron.calculate_error_hidden(hidden_neuron.output, prev_weights, prev_errors) # Error berekenen

                """Voor de update functie moet een neuron de output van elke neuron in de vorige laag weten,
                hier zet ik dit in een lijst zodat we dit mee kunnen geven"""
                for layer_index, layer in enumerate(reversed_layers):
                    if layer_index > len(self.layers)-2:
                        prev_outputs = input_list
                    else:
                        prev_outputs = []
                        for neuron in reversed_layers[layer_index+1].neurons:
                            prev_outputs.append(neuron.output)
                    """Als laaste loopen we door alle neurons in de laag heen om deze te updaten."""
                    for neuron in layer.neurons:
                        neuron.update(prev_outputs, learning_rate)

    def __str__(self):
        return f'----------------------------------------\n' \
               f'Dit netwerk bestaat uit {len(self.layers)} layers\n'
