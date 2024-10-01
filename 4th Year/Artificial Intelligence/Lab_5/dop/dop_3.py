def can_form_palindrome(n):
    digit_count = {}
    for digit in str(n):
        digit_count[digit] = digit_count.get(digit, 0) + 1
    odd_count = sum(1 for count in digit_count.values() if count % 2 != 0)
    return odd_count <= 1


number = int(input("Введите натуральное число: "))
if can_form_palindrome(number):
    print("Да, можно создать палиндром.")
else:
    print("Нет, нельзя создать палиндром.")