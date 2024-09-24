#include <iostream>
#include <fstream>
#include <queue>

using namespace std;

int main()
{  
    ifstream fin("huffman.in");
    ofstream fout("huffman.out");
    int n;
    fin >> n;
    long long sum = 0;
     long long* mas = new long long[n];
    for(int i = 0; i < n; i++)
    {
        fin >> mas[i];
    }
    priority_queue < long long> q;
    for(int i = 0; i < n; i++)
    {
        q.push(-1 * mas[i]);
    } 
     long long x;
     long long y;
     long long temp;
    while(q.size() != 1)
    {
        x = q.top();
        q.pop();
        y = q.top();
        q.pop();
        temp = x+y;
        sum += temp;
        q.push(temp);
    }  
    fout << -1 *sum;
}