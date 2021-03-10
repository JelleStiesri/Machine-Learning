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

                # for layer in self.layers:
                #     for neuron in layer.neurons:
                #         #OUTPUT VORIGE LAYER
                #         neuron.update(neuron.output)
                # H1 H2 H3 Output
                # Output H3 H2 H1

                for layer_index, layer in enumerate(reversed_layers):
                    # print('\n')
                    if layer_index > len(self.layers)-2:
                        # print('ohjee')
                        # print(f'Index {layer_index}')
                        prev_outputs = input_list
                    else:
                        # print(layer_index, layer.name)
                        # print(reversed_layers[layer_index+1].name)
                        prev_outputs = []
                        for neuron in reversed_layers[layer_index+1].neurons:
                            prev_outputs.append(neuron.output)
                    # print(f'prevlayer: {prev_outputs}\n')



                    for neuron in layer.neurons:
                        # print(prev_outputs)
                        neuron.update(prev_outputs, learning_rate)












    def __str__(self):
        return f'----------------------------------------\n' \
               f'Dit netwerk bestaat uit {len(self.layers)} layers\n'
