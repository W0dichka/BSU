surnames = ["Канашевич", "Криворотов", "Бекетов", "Лукин", "Налетько", "Статкевич", "Рудинский", "Козлов", "Еда", "Слук"]
names = ['Кирилл', 'Женя', 'Дима', 'Антон', 'Арина', 'Захар','Егор', 'Влад', 'Никита', 'Женя']

with open('full_names.txt', 'w', encoding='utf-8') as file:
    for surname, name in zip(surnames, names):
        file.write(f"{surname} {name}\n")

with open('full_names.txt', 'r', encoding='utf-8') as file:
    for line in file:
        print(line.strip())