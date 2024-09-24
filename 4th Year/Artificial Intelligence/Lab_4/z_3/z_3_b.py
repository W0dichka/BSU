n = int(input("Введите количество пар синонимов: "))
synonyms = {}

for _ in range(n):
    word1, word2 = input().split()
    synonyms[word1] = word2
    synonyms[word2] = word1

word_to_find = input("Введите слово для поиска синонима: ")

if word_to_find in synonyms:
    print(synonyms[word_to_find])
else:
    print("Слово не найдено в словаре.")