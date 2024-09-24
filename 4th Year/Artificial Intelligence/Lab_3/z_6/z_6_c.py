def letter_frequency(s):
    frequency = {}
    for char in s:
        if char.isalpha():
            char = char.lower()
            if char in frequency:
                frequency[char] += 1
            else:
                frequency[char] = 1

    return frequency

input_string = input()
result = letter_frequency(input_string)
print(result)