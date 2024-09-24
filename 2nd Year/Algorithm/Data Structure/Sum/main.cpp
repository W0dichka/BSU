#include <iostream>
#include <string>

using namespace std;

int main()
{  
    int n;
    cin >> n;
    long long** mas = new long long*[n+1];
    for(int i = 0; i < n+1; i++)
    {
        mas[i] = new long long[2];
    }
    cin >> mas[0][0];
    mas[0][1] = 0;
    mas[1][1] = mas[0][0];
    
    for(int i = 1; i < n; i++)
    {
        cin >> mas[i][0];
        mas[i+1][1] = mas[i][0]+mas[i][1];
    }
    int q;
    string s;
    cin >> q;
    int index1, index2;
    long long x;
    while(q !=0)
    {
        cin >> s;
        if(s == "FindSum")
        {
            cin >> index1 >> index2;
            cout << mas[index2][1] - mas[index1][1]<<endl;
        }
        else
        {
            cin >> index1 >> x;
            mas[index1][0] += x;
            for (int i = index1; i < n;i++)
            {
                mas[index1+1][1] += x;
            }
        }
        q--;
   }
}