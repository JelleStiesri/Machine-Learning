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

    def train(self, dataSet, epochs):
        for epoch in range(epochs):
            random.shuffle(dataSet)

            inputs = []
            targets = []

            for line in dataSet:
                inputs.append(line[0])
                targets.append(line[1])

            # print('INPUTS', inputs, targets)


            for index, input_list in enumerate(inputs):

                self.feed_forward(input_list)  # Voert het hele netwerk uit zodat de errors&delta's berekend kunnen worden
                target = targets[index]
                # print(index, input_list, target)


                #OUTPUT NEURONS
                for index, output_neuron in enumerate(self.layers[-1].neurons):
                    # print(output_neuron)
                    # print(output_neuron.calculate_error_output(output_neuron.output, target))
                    output_neuron.calculate_error_output(output_neuron.output, target[index])
                    # print('ERROR output neuron berekend\n')
                    # output_neuron.update(output_neuron.output)
                    # print(output_neuron)

                #HIDDEN NEURONS
                reversed_layers = list(reversed(self.layers)) # Reversed omdat we van achter naar voren werken

                for layer_index, layer in enumerate(reversed_layers[1:]):  # Hierin nemen we niet de output layer mee
                    # print("\nVOOR LAAG IN HIDDEN LAGEN ")
                    # print(f'LAYER_INDEX: {layer_index} - LAYER NAME: {layer.name}')
                    for neuron_index, hidden_neuron in enumerate(layer.neurons):
                        # print('\n\n\n\nVOOR HIDDEN NEURON IN LAGEN')
                        # print('NEURON_INDEX: ',neuron_index, hidden_neuron)
                        # VORIGE GEWICHTEN (juiste index!)



                        prev_errors = []
                        prev_weights = []
                        for neuron in reversed_layers[layer_index].neurons:
                            prev_weights.append(neuron.weights[neuron_index])
                            prev_errors.append(neuron.error)
                        # print(f'prev_error: {prev_errors}\nprev_weights: {prev_weights}')
                        hidden_neuron.calculate_error_hidden(hidden_neuron.output, prev_weights, prev_errors)
                        # print('ERROIR HIDDEN LAYEr:',hidden_neuron.calculate_error_hidden(hidden_neuron.output, prev_weights, prev_errors ))



                for layer in self.layers:
                    for neuron in layer.neurons:
                        neuron.update(neuron.output)












    def __str__(self):
        return f'----------------------------------------\n' \
               f'Dit netwerk bestaat uit {len(self.layers)} layers\n'
