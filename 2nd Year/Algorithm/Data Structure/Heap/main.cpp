#include <iostream>
#include <fstream>
#include <climits>

using namespace std;

int is_heap(int n, int mas[])
{
    for (int i = 1; i < n + 1; i++)
    {
        if (mas[i] <= mas[2 * i] && mas[i] <= mas[2 * i + 1])
        {

        }
        else
        {
            return 0;
        }
    }
    return 1;
}


int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int n;
    fin >> n;
    int* mas = new int[3*n];
    for (int i = 0; i < 3*n; i++)
    {
        mas[i] = INT_MAX;
    }
    for (int i = 1; i < n + 1; i++)
    {
        fin >> mas[i];
    }
    if (n == 1)
    {
        fout << "Yes";
        return 0;
    }
    if (is_heap(n , mas) == 1)
    {
        fout << "Yes";
    }
    else
    {
        fout << "No";
    }
}
