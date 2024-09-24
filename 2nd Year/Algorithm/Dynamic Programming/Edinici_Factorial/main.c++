#include <iostream>

using namespace std;

const long long MOD = 1e9 + 7; 

long long bin_pow(long long x, long long y) {
    if (y == 1) {
        return x;
    }

    if (y % 2 == 0) {
        long long t = bin_pow(x, y / 2);
        return t * t % MOD;
    } else {
        return bin_pow(x, y - 1) * x % MOD;
    }
}

long long inverse_element(long long x) {
    return bin_pow(x, MOD - 2);
}

long long divide(long long x, long long y) {
    return x * inverse_element(y) % MOD;
}

long long factorial(long long n)
{
    long long fact = 1;
    for(long long i = 2; i <= n; i++)
    {
        fact = fact * i % MOD;
    }
    return fact;
}

long long sochetanie(long long n, long long k)
{
    return divide(divide(factorial(n),factorial(k)),factorial(n-k));
}

int main()
{
    long long n;
    long long k;
    cin >> n >> k;
    long long answer = sochetanie(n,k);
    cout << answer << endl;
}