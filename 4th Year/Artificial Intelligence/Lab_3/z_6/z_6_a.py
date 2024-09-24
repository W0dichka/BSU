def count_vowels_and_consonants(s):
    vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
    consonants = "бвгджзйклмнпрстфхцчшщБВГДЖЗЙКЛМНПРСТФХЦЧШЩ"
    vowel_count = 0
    consonant_count = 0

    for char in s:
        if char in vowels:
            vowel_count += 1
        elif char in consonants:
            consonant_count += 1

    return vowel_count, consonant_count

input_string = input()
vowel_count, consonant_count = count_vowels_and_consonants(input_string)
print(f"Гласные: {vowel_count}, Согласные: {consonant_count}")