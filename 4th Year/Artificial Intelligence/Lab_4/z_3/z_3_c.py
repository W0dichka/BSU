n = int(input("Введите количество строк: "))
word_count = {}

for _ in range(n):
    line = input().strip()
    words = line.split()
    for word in words:
        word = word.lower()
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

max_count = 0
result_word = None

for word, count in word_count.items():
    if count > max_count or (count == max_count and (result_word is None or word < result_word)):
        max_count = count
        result_word = word

print(result_word)