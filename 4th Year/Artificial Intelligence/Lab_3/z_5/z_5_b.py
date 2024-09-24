def greater_than_previous(lst):
    result = []
    for i in range(1, len(lst)):
        if lst[i] > lst[i - 1]:
            result.append(lst[i])
    return result

input_list = [1, 3, 2, 5, 4, 6]
result = greater_than_previous(input_list)
print(result)