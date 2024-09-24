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

count_word_pairs = [(count, word) for word, count in word_count.items()]
count_word_pairs.sort(key=lambda x: (-x[0], x[1]))

for count, word in count_word_pairs:
    print(word)