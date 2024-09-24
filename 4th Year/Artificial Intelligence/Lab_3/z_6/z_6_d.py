def count_same_start_end_words(s):
    words = s.split()
    count = 0
    for word in words:
        if word and word[0].lower() == word[-1].lower():
            count += 1

    return count

input_string = input()
result = count_same_start_end_words(input_string)
print(result)