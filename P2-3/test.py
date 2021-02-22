from Perceptron import *
import pandas as pd


def create_table_data(function, expectations):
    """Deze functie berekent de uitkomst van een perceptron/netwerk, Als input gebruikt hij de door mij gegeven data
    en haalt daar de uitkomst vanaf (zodat de inputs hetzelfde zijn)"""
    data = []

    for item in expectations:

        output = function(item[0])
        data.append([item[0], output])
    return data

def print_table(data, caption):
    """Deze functie print een waarheidstabel"""

    columns = ['Input', 'Output']

    # for item in range(len(data[0])):
    #     columns.append('x'+str(item+1))
    # for item in range(len(data[1])):
    #     columns.append('Output')



    df = pd.DataFrame.from_records(data, columns=columns)
    print(f'<{caption}>\n {df}\n')

p = Perceptron('AND', 4, [0.2, 0.1])
# p = Perceptron('AND', -1, [0.5, 0.5])
expectations = [[[False, False], False],
                [[False, True], False],
                [[True, False], False],
                [[True, True], True]]


print_table(expectations, 'Verwachting')
print_table(create_table_data(p.activation, expectations), 'Uitkomst')
print(f'Weights: {p.weights} - Bias: {p.bias}')
print(f'Error = {p.error(expectations)}')



p.update(expectations, 500)
print_table(create_table_data(p.activation, expectations), 'Uitkomst')
print(f'Weights: {p.weights} - Bias: {p.bias}')
print(f'Error = {p.error(expectations)}')