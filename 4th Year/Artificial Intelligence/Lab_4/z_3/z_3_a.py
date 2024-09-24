text = input("Введите текст: ")

words = text.split()
word_count = {}
result = []
for word in words:
    if word in word_count:
        result.append(word_count[word]) 
        word_count[word] += 1 
    else:
        result.append(0)  
        word_count[word] = 1

print(" ".join(map(str, result)))