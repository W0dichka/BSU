with open('full_names.txt', 'r', encoding='utf-8') as file:
    for line in file:
        surname, name = line.strip().split()
        print(f"{surname} {name[0]}.")