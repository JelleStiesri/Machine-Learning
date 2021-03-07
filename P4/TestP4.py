from Neuron import *
import random
import pandas as pd

def create_table_data(function, expectations):
    """Deze functie berekent de uitkomst van een perceptron/netwerk en geeft dit terug in een goed format"""
    data = []

    for item in expectations:

        output = function(item[0])
        data.append([item[0], output])
    return data


def print_table(data, caption):
    """Deze functie print een waarheidstabel"""

    columns = ['Input', 'Output']

    df = pd.DataFrame.from_records(data, columns=columns)
    print(f'<{caption}>\n {df}\n')


N1 = Neuron('Neuron 1', 'Output', -1.2, [0.5, 0.1])

AND_expectations = [[[False, False], False],
                    [[False, True], False],
                    [[True, False], False],
                    [[True, True], True]]

# print('Verwachting', AND_expectations[1])
print(N1)

print_table(AND_expectations, "Verwachting")

updated_output = create_table_data(N1.activation, AND_expectations)
print_table(updated_output, 'Uitkomst voor training')



for i in range(10000):
    random.shuffle(AND_expectations)
    for input_list, expectation in AND_expectations:

        output = N1.activation(input_list)
        # print(f'input: {input_list, expectation} --- Output: {output}')
        N1.calculate_error(output, expectation)
        # print('error ',N1.error)
        N1.update(output)

        # print('\n')

updated_output = create_table_data(N1.activation, AND_expectations)
print_table(updated_output, 'Uitkomst Na Training')
