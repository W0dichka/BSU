#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int max(int a, int b)
{
    if (a > b)
    {
        return a;
    }
    return b;
}
int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int n;
    fin >> n;
    vector<int> mas(n);
    vector<int> sum(n);
    if (n == 2)
    {
        fout << -1;
        return 0;
    }
    for (int i = 0; i < n; i++)
    {
        fin >> mas[i];
    }
    if (n == 1)
    {
        fout << mas[0];
        return 0;
    }
    sum[1] = -1;
    sum[0] = mas[0];
    sum[2] = sum[0] + mas[2];
    for (int i = 3; i < n; i++)
    {
        if ((sum[i - 3] == -1) && (sum[i - 2] == -1))
        {
            sum[i] = -1;
        }
        sum[i] = max(sum[i - 2], sum[i - 3]) + mas[i];
    }
    fout << sum[n - 1];
}
