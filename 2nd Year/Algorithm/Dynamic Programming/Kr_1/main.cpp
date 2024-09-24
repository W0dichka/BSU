#include <iostream>
#include <fstream>
#include <climits>

using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int n, m;
	fin >> n >> m;
	int** matrix = new int* [n+2];
	for (int i = 0; i < n+2; i++)
	{
		matrix[i] = new int[m+2];
	}
	for (int i = 0; i < n + 2; i++)
	{
		for (int j = 0; j < m + 2; j++)
		{
			matrix[i][j] =INT_MAX;
		}
	}
	for (int i = 1; i < n+1; i++)
	{
		for (int j = 1; j < m+1; j++)
		{
			fin >> matrix[i][j];
		}
	}
	//
	int ans = INT_MAX;
	int sum = 0;
	int j;
	//sum;
    if (m == 1)
    {
        for(int i = 1; i < n + 1; i++)
        {
            sum += matrix[i][1];
        }
        fout << sum;
        return 0;
    }
	for (int i = n-1; i > 0; i--)
	{
        for(int j = 1; j < m + 1; j++)
        {
            matrix[i][j] += min(matrix[i+1][j-1], min(matrix[i+1][j],matrix[i+1][j+1]));
        }
	}
    for(int j = 1; j < m + 1; j++)
    {
        if(matrix[1][j] < ans)
        {
            ans = matrix[1][j];
        }
    }
    fout << ans;
}