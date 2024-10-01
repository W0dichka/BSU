with open('surnames.txt', 'r', encoding='utf-8') as file:
    surnames = file.readlines()

for index, surname in enumerate(surnames, start=1):
    print(f"{index}: {surname.strip()}")