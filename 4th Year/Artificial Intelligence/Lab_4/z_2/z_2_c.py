N, M = map(int, input().split())

anya_colors = set(int(input()) for _ in range(N))
borya_colors = set(int(input()) for _ in range(M))

common_colors = anya_colors & borya_colors
anya_only_colors = anya_colors - borya_colors
borya_only_colors = borya_colors - anya_colors

def print_result(label, color_set):
    sorted_colors = sorted(color_set)
    print(f"{label}:")
    print(len(sorted_colors))
    if sorted_colors:
        print(" ".join(map(str, sorted_colors)))
    else:
        print("Нет")

print_result("Common", common_colors)
print_result("Anya",anya_only_colors)
print_result("Borya",borya_only_colors)