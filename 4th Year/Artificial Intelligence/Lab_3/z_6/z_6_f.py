def rearrange_list(lst):
    non_zero = [x for x in lst if x != 0]
    zeroes = [x for x in lst if x == 0]
    return non_zero + zeroes

input_list = [0, 1, 0, 2, 3, 0, 4]
result = rearrange_list(input_list)
print(result)