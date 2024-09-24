import random

def generate_random_list(size, lower_bound, upper_bound):
    return [random.randint(lower_bound, upper_bound) for _ in range(size)]

def find_max_min_indices(lst):
    if not lst:
        return None, []
    max_value = max(lst)
    min_value = min(lst)
    max_indices = [i for i, x in enumerate(lst) if x == max_value]
    min_indices = [i for i, x in enumerate(lst) if x == min_value]
    return (max_value, max_indices), (min_value, min_indices)

random_list = generate_random_list(10, 1, 100)
print("Список:", random_list)

(max_value_info, min_value_info) = find_max_min_indices(random_list)
print(f"Максимальный элемент: {max_value_info[0]}, Индексы: {max_value_info[1]}")
print(f"Минимальный элемент: {min_value_info[0]}, Индексы: {min_value_info[1]}")