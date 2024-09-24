def is_palindrome(word):
    return word == word[::-1]

input_string = input("Введите строку: ")
words = input_string.split()
longest_palindrome = ""
for word in words:
    if is_palindrome(word):
        if len(word) > len(longest_palindrome):
            longest_palindrome = word
if longest_palindrome:
    print("Самый длинный перевертыш:", longest_palindrome)
else:
    print("Перевертышей не найдено.")