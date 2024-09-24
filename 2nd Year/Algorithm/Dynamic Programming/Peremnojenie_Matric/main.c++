#include <iostream>
#include <fstream>
#include <climits>

using namespace std;	

int main()
{
	ifstream fin("input.txt");
	ofstream fout("output.txt"); 
	int n;
	fin >> n;
	int** matrix;
	matrix = new int* [n];
	int* dim = new int[n+1];
	for (int i = 0; i < n; i++) 
	{
		matrix[i] = new int[n];
	}
	fin >> dim[0];
	int x;
	for (int i = 1; i < n+1; i++)
	{
		fin >> dim[i];
		fin >> x;
	}
	fin >> dim[n];

	for (int i = 0; i < n; i++)
	{
		matrix[i][i] = 0;
	}
	for (int l = 1; l < n; l++)
	{
		for (int i = 0; i < n - l; i++)
		{
			int j = i + l;

			matrix[i][j] = INT_MAX;

			for (int k = i; k < j; k++)
			{
				int cost = matrix[i][k] + matrix[k + 1][j] + dim[i] * dim[k + 1] * dim[j + 1];

				if (cost < matrix[i][j])
					matrix[i][j] = cost;
			}
		}
	}
	fout << matrix[0][n - 1];
}
