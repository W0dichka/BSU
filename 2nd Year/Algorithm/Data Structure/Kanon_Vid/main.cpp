#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int n,u,v;
    fin >> n;
    int* mas = new int[n+1];
    for (int i = 0; i < n+1; i++)
    {
        mas[i] = 0;
    }
    for (int i = 0; i < n; i++)
    {
        fin >> u >> v;
        mas[v] = u;
    }
    fout << mas[1];
    for (int i = 2; i < n+1; i++)
    {
        fout << " " << mas[i];
    }
}
