def extract_and_sort_names(s):
    names = {word for word in s.split() if word[0].isupper()}
    return sorted(names)

input_string = input()
result = extract_and_sort_names(input_string)
print(result)