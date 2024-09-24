#include <iostream>
#include <fstream>
#include <string> 

using namespace std;

int main()
{
    ifstream fin("in.txt");
    ofstream fout("out.txt");
    int del;
    int ins;
    int er;
    string s1;
    string s2;
    fin >> del >> ins >> er;
    fin >> s1 >> s2;
    int n = s1.size();
    int m = s2.size();
    int** matrix = new int* [n+1];
    for (int i = 0; i < n+1; i++)
    {
        matrix[i] = new int[m+1];
    }
    matrix[0][0] = 0;
    for (int i = 1; i < n+1; i++)
    {
        matrix[i][0] = i * del;
    }
    for (int i = 1; i < m+1; i++)
    {
        matrix[0][i] = i * ins;
    }
    int temp = 1;
    for (int i = 1; i < n + 1; i++)
    {
        for (int j = 1; j < m + 1; j++)
        {
            if (s1[i-1] == s2[j-1]) {
                temp = 0;
            }
            matrix[i][j] = min(min(matrix[i - 1][j] + del, matrix[i][j - 1] + ins), matrix[i - 1][j - 1] + er * temp);
            temp = 1;
        }
    }
    fout << matrix[n][m];
}
