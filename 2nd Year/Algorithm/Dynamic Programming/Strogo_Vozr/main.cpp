#include <iostream>
#include <fstream>
#include <climits>
#include <algorithm>
#include <vector>

using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    int n;
    fin >> n;
    int* mas = new int[n];
    int length = 0;
    if( n == 0)
    {
        fout << "0";
        return 0;
    }

    for(int i = 0; i < n; i++)
    {
        fin >> mas[i];
    }

    if(n == 1)
    {
        fout << 1;
        return 0;
    }

    vector<int> posl(n+1);

    for( int i = 1; i < n + 1; i++)
    {
        posl[i] = INT_MAX;
    }
    posl [0] = INT_MIN;

    for(int i = 0; i < n; i++)
    {
        int key = int(upper_bound(posl.begin(), posl.end(), mas[i]) - posl.begin());
        if(mas[i] < posl[key] && posl[key-1] < mas[i])
        {
            posl[key] = mas[i];
            if(key > length)
            {
                length = key;
            }
        }
    }
    fout << length;
}