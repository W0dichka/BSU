def capitalize(word):
    if not word:
        return word
    return chr(ord(word[0]) - 32) + word[1:]


input_string = input("Введите строку: ")

capitalized_words = [capitalize(word) for word in input_string.split()]
output_string = ' '.join(capitalized_words)
print(output_string)