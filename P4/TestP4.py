from Neuron import *
import random

N1 = Neuron('Neuron 1', -1.2, [0.5, 0.1])

AND_expectations = [[[False, False], False],
                    [[False, True], False],
                    [[True, False], False],
                    [[True, True], True]]

print('Verwachting', AND_expectations[1])
print(N1)
# print('Activation:', N1.activation(AND_expectations[1][0]))

# N1.calculate_error(AND_expectations[1])
# N1.update(AND_expectations)

# random.shuffle(AND_expectations)
# print(AND_expectations)

for i in range(1):
    for input_list, expectation in AND_expectations:

        activation = N1.activation(input_list)
        print(f'input: {input_list, expectation} --- Output: {activation}')
        N1.calculate_error(activation, expectation)
        print('error ',N1.error)
        N1.update()

        print('\n')
