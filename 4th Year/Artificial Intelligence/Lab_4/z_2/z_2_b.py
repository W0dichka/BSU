def chek(list):
    seen = set()
    for el in list:
        if el in seen:
            print(el, "Yes")
        else:
            seen.add(el)
            print(el, "No")
        


list = [1, 2, 4, 2, 5, 1, 8]

chek(list)
