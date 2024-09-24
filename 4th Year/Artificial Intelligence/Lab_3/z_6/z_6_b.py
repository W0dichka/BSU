def format_sentence(sentence):
    words = sentence[:-1].split()
    formatted_words = [word.capitalize() + '.' for word in words]
    return ' '.join(formatted_words)

input_sentence = input()
result = format_sentence(input_sentence)
print(result)