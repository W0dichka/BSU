def count_words(s):
    s = s.strip()
    if not s:
        return 0
    return s.count(' ') + 1

input_string = input()
word_count = count_words(input_string)
print(f"Количество слов: {word_count}")