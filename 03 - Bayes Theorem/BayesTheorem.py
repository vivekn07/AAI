# Calculate the probability of cancer patient and diagnostic test
def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):

    # Calculate P(not A)
    not_a = 1 - p_a

    # Calculate P(B)
    p_b = p_b_given_a * p_a + p_b_given_not_a * not_a

    # Calculate P(A|B)
    p_a_given_b = (p_b_given_a * p_a) / p_b
    return p_a_given_b

# P(A)
p_a = 0.0002

# P(B|A)
p_b_given_a = 0.85

# P(B|not A)
p_b_given_not_a = 0.05

# Calculate P(A|B)
result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)

# Summarize and printing the result
print(f'P(A|B) = {result * 100:.3f}%')