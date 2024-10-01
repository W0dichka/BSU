def remove_duplicate_words(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        words = content.split()
        unique_words = list(dict.fromkeys(words))
        result = ' '.join(unique_words)
        print(result)
    except FileNotFoundError:
        print("Файл не найден. Пожалуйста, проверьте название файла.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

filename = input("Введите название текстового файла: ")
remove_duplicate_words(filename)