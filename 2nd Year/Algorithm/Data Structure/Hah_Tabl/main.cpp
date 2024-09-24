#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int c, m, n, x, index, j;
    j = 0;
    fin >> m >> c >> n;
    int* mas = new int[m];
    vector <bool> unique(1000000001, false);
    for (int i = 0; i < m; i++)
    {
        mas[i] = -1;
    }
    for (int i = 0; i < n; i++) 
    {  
        fin >> x;
        if (unique[x] == false)
        {
            unique[x] = true;
            index = ((x % m) + (c * j)) % m;
            j++;
            while (mas[index] != -1)
            {
                index = ((x % m) + (c * j)) % m;
                j++;
            }
            mas[index] = x;
            j = 0;
        }
    }
    fout << mas[0];
    for (int i = 1; i < m; i++)
    {
        fout << " " << mas[i];
    }
}
