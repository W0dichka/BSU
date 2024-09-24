def count_greater_than_neighbors(lst):
    count = 0
    for i in range(1, len(lst) - 1):
        if lst[i] > lst[i - 1] and lst[i] > lst[i + 1]:
            count += 1
    return count

input_list = [1, 3, 2, 5, 4, 6]
result = count_greater_than_neighbors(input_list)
print(result)