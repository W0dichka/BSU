#include <iostream>
#include <fstream>
#include <cmath>

void dec_to_bin(unsigned long long n, int mas[])
{
    int i = 0;
    while(n)
    {
        mas[i] = n % 2;
        n /=2;
        i++;
    }
}


using namespace std;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    unsigned long long n;
    fin >> n;
    int size = (log10(n) / log10(2)) + 1;
    int* mas = new int[size];
    dec_to_bin(n, mas);
    for(int i = 0; i < size; i++)
    {
        if(mas[i] == 1)
        {
            fout << i <<endl;
        }
    }
}