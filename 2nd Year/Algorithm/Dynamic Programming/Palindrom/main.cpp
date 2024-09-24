#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    string s;
    fin >> s;
    int n = s.length();
    int ** matrix = new int* [n];
    for(int i = 0; i < n; i++)
    {
        matrix[i] = new int[n];
    }
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            matrix[i][j] = 0;
        }
    }
    for(int i = 0; i < n; i++)
    {
        matrix[i][i] = 1;
    }
    for(int i = 1; i < 2; i++)
    {
        for(int j = 0; j < n; j++)
        {
            if(s[j] == s[j+1])
                matrix[j][j+i] = 2;
            else    
                matrix[j][j+i] = 1;
        }
    }
    for(int j = 2; j < n; j++)
    {
        for(int i = 0; i < n-j; i++)
        {
            if(s[i] == s[j+i])
            {
                matrix[i][j+i] = matrix[i+1][j+i-1] + 2;
            }
            else    
            {
                matrix[i][j+i] = max( matrix[i+1][j+i], matrix[i][j+i-1] );
            }
        }
    }
    int len = matrix[0][n-1];
    if(len == 1)
    {
        fout << 1 << endl;
        fout << s[0];
        return 0;
    }
    fout << len <<endl;
    char* pal = new char[n + 1];
    int i=0;
    int j = n-1;
    int k=0;
    while( j >= i)
    {
        while(matrix[i][j] == matrix[i+1][j] && i < n-1)
        {
            i++;
        }
        while(matrix[i][j] == matrix[i][j-1] && j > 0)
        {
            j--;
        }
        pal[k] = s[j];
        k++;
        i++;
        j--;
    }
    if(len % 2 == 0)
    {
        for(int i = 0; i < k;i++)
        {
            fout << pal[i];
        }
        for(int i = k-1; i >= 0; i--)
        {
            fout << pal[i];
        }
    }
    else
    {
        for(int i = 0; i < k;i++)
        {
            fout << pal[i];
        }
        for(int i = k-2; i >= 0; i--)
        {
            fout << pal[i];
        }
    }
}