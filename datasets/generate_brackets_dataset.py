from random import randint, choice
import json
import jsonlines

def generate_correct_bracket_sequence(n):
    # Generates a correct bracket sequence of length 2 * n
    if n == 0:
        return ""
    A_part_size = randint(0, n - 1)
    B_part_size = n - A_part_size - 1
    A_part = generate_correct_bracket_sequence(A_part_size)
    B_part = generate_correct_bracket_sequence(B_part_size)
    return "(" + A_part + ")" + B_part

def generate_incorrect_bracket_sequence(n):
    # Generates an incorrect bracket sequence of length 2 * n
    while True:
        sequence = ''.join(choice(['(', ')']) for _ in range(2 * n))
        if is_correct_bracket_sequence(sequence):
            continue
        return sequence
    
def is_correct_bracket_sequence(sequence):
    # Checks if the given bracket sequence is correct
    balance = 0
    for bracket in sequence:
        if bracket == '(':
            balance += 1
        else:
            balance -= 1
        if balance < 0:
            return False
    return balance == 0

def generate_bracket_dataset(max_size, num_examples, output_file):
    # generates a dataset of bracket sequences of size max_size * 2 * num_examples:
    # for each size from 1 to max_size, generates num_examples correct bracket sequences 
    # and num_examples incorrect bracket sequences
    
    data = []
    labels = []

    for n in range(1, max_size + 1):
        for _ in range(num_examples):
            data.append(generate_correct_bracket_sequence(n))
            labels.append('correct')
            data.append(generate_incorrect_bracket_sequence(n))
            labels.append('incorrect')

    with jsonlines.open(output_file, "w") as writer:
        for i in range(len(data)):
            writer.write({"input": ' '.join(data[i]), "label": labels[i]})

if __name__ == "__main__":
    generate_bracket_dataset(128, 1000, "../data/train_brackets_dataset.json")
    generate_bracket_dataset(128, 100, "../data/test_brackets_dataset.json")
