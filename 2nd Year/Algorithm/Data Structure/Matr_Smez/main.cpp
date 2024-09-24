#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int n, m,u,v;
    fin >> n >> m;
    int mas[101][101];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            mas[i][j] = 0;
        }
    }
    for (int i = 0; i < m; i++)
    {
        fin >> u;
        fin >> v;
        mas[u - 1][v - 1] = 1;
        mas[v - 1][u - 1] = 1;
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            fout << mas[i][j] << " ";
        }
        fout << endl;
    }
}
