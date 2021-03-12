from Neuron import *
import random

class NeuronNetwork:

    def __init__(self, layers):
        self.layers = layers

    def feed_forward(self, input_list):
        """Loopt door de layers heen en geeft de output van de vorige
        als input voor de volgende."""
        for index in range(len(self.layers)):
            print("jpoi")

            layer = self.layers[index]  # selecteerde de layer
            input_list = layer.activate(input_list)
        return input_list

    def train(self, dataSet, epochs, learning_rate):
        for epoch in range(epochs):
            random.shuffle(dataSet)

            inputs = []
            targets = []

            for line in dataSet:
                inputs.append(line[0])
                targets.append(line[1])

            for index, input_list in enumerate(inputs):

                self.feed_forward(input_list)  # Voert het hele netwerk uit zodat de errors&delta's berekend kunnen worden
                target = targets[index]

                #OUTPUT NEURONS
                for index, output_neuron in enumerate(self.layers[-1].neurons):
                    output_neuron.calculate_error_output(output_neuron.output, target[index])

                #HIDDEN NEURONS
                reversed_layers = list(reversed(self.layers)) # Reversed omdat we van achter naar voren werken

                for layer_index, layer in enumerate(reversed_layers[1:]):  # Hierin nemen we niet de output layer mee
                    for neuron_index, hidden_neuron in enumerate(layer.neurons):
                        prev_errors = []
                        prev_weights = []
                        for neuron in reversed_layers[layer_index].neurons:
                            prev_weights.append(neuron.weights[neuron_index])
                            prev_errors.append(neuron.error)

                        hidden_neuron.calculate_error_hidden(hidden_neuron.output, prev_weights, prev_errors)



                for layer_index, layer in enumerate(reversed_layers):
                    if layer_index > len(self.layers)-2:
                        prev_outputs = input_list
                    else:
                        prev_outputs = []
                        for neuron in reversed_layers[layer_index+1].neurons:
                            prev_outputs.append(neuron.output)



                    for neuron in layer.neurons:

                        neuron.update(prev_outputs, learning_rate)











    def __str__(self):
        return f'----------------------------------------\n' \
               f'Dit netwerk bestaat uit {len(self.layers)} layers\n'
