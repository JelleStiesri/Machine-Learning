def calculate_derivative(output: float):
    return output * (1 - output)  # σ'(inputj) = σ(inputj) ∙ (1 – σ(inputj)) = outputj ∙ (1 – outputj)


def calculate_error_output(output: float, target: float):
    error = calculate_derivative(output) * -(target - output)  # Δj = σ'(inputj) ∙ –(targetj – outputj)

    # self.error = error
    return error


# print(calculate_error_output(0.9999379249875231, 0.0))

# print(calculate_derivative(0.9999797974098636))
numb = 0.9999797974098636

print(1-numb)
pop = 1-numb
print(numb*pop)