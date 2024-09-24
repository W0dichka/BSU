def swap_words(s):
    return ' '.join(s.split()[::-1])

input_string = input()
result = swap_words(input_string)
print(result)