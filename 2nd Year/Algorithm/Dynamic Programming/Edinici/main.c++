#include <iostream>

using namespace std;

const long long MOD = 1e9 + 7; 

int sochetanie(int n, int k)
{
    int** mas = new int* [n + 1];
    for (int i = 0; i < n + 1; i++)
    {
        mas[i] = new int[n + 1];
    }
    for (int i = 0; i < n + 1; i++)
    {
        for (int j = 0; j < n + 1; j++)
        {
            mas[i][j] = 0;
        }
    }
    for(int i =0 ;i < n+1;i++)
    {
        mas[i][0] = 1;
        mas[i][i] = 1;
        for(int j = 1; j < i; j++)
        {
            mas[i][j] = mas[i-1][j-1]%  MOD + mas[i-1][j] % MOD;
        }
    }
    return mas[n][k] % MOD;
}

int main()
{
    int n;
    int k;
    cin >> n >> k;
    int answer = sochetanie(n,k);
    cout << answer << endl;
}