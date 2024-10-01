def custom_max(*args):
    if not args:
        raise ValueError("Функция max() требует хотя бы один аргумент")
    maximum = args[0] 
    for num in args:
        if num > maximum:
            maximum = num 
    return maximum

print(custom_max(1, 5, 3, 9, 2))  
print(custom_max(-1, -5, -3)) 
print(custom_max(7))