surnames = ["Канашевич", "Криворотов", "Бекетов", "Лукин", "Налетько", "Статкевич", "Рудинский", "Козлов", "Еда", "Слук"]

with open("surnames.txt", "w", encoding="utf-8") as file:
    for surname in surnames:
        file.write(surname + "\n")