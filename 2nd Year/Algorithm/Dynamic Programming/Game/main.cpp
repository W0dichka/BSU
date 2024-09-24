#include <iostream>
#include <fstream>
#include <stack>

using namespace std;

int mas[13][13][13][13][13][13][13];
int win = 0;

int main()
{
    ifstream fin("input.txt");
    ofstream fout("output.txt");
    long long n;
    fin >> n;

    //  0
    for (int i1 = 10; i1 < 13; i1++)
	{
		for (int i2 = 10; i2 < 13; i2++)
		{
			for (int i3 = 10; i3 < 13; i3++)
			{
				for (int i4 = 10; i4 < 13; i4++)
				{
					for (int i5 = 10; i5 < 13; i5++)
					{
						for (int i6 = 10; i6 < 13; i6++)
						{
							for (int i7 = 10; i7 < 13; i7++)
							{
                                mas[i1][i2][i3][i4][i5][i6][i7] = 0;
							}
						}
					}
				}
			}
		}
	}
    mas[9][9][9][9][9][9][9] = 0; 
    mas[9][9][9][9][9][9][12] = 1;
    //win
    for (int i1 = 9; i1 >= 0; i1--)
	{
		for (int i2 = 9; i2 >= 0; i2--)
		{
			for (int i3 = 9; i3 >= 0; i3--)
			{
				for (int i4 = 9; i4 >= 0; i4--)
				{
					for (int i5 = 9; i5 >= 0; i5--)
					{
						for (int i6 = 9; i6 >= 0; i6--)
						{
							for (int i7 = 9; i7 >= 0; i7--)
							{
                                win = 0;
                                for(int k = 1; k < 4; k++)
                                {
                                    if(mas[i1][i2][i3][i4][i5][i6][i7+k] != 0)
                                    {
                                        win++;
                                    }
                                    if(mas[i1][i2][i3][i4][i5][i6+k][i7] != 0)
                                    {
                                        win++;
                                    }
                                    if(mas[i1][i2][i3][i4][i5+k][i6][i7] != 0)
                                    {
                                        win++;
                                    }
                                    if(mas[i1][i2][i3][i4+k][i5][i6][i7] != 0)
                                    {
                                        win++;
                                    }
                                    if(mas[i1][i2][i3+k][i4][i5][i6][i7] != 0)
                                    {
                                        win++;
                                    }
                                    if(mas[i1][i2+k][i3][i4][i5][i6][i7] != 0)
                                    {
                                        win++;
                                    }
                                    if(mas[i1+k][i2][i3][i4][i5][i6][i7] != 0)
                                    {
                                        win++;
                                    } 
                                }
                                if(win == 0)
                                {
								    mas[i1][i2][i3][i4][i5][i6][i7] = 1;
                                }  
                                else
                                {
								    mas[i1][i2][i3][i4][i5][i6][i7] = 0;
                                }  
                                mas[9][9][9][9][9][9][9] = 0;                                                                                                                                                                                                                  
							}
						}
					}
				}
			}
		}
	}
    //answer
    int i1,i2,i3,i4,i5,i6,i7;
    i1 = n % 10000000 / 1000000;
    i2 = n % 1000000 / 100000;
    i3 = n % 100000 / 10000;
    i4 = n % 10000 / 1000;
    i5 = n % 1000 / 100;
    i6 = n % 100 / 10;
    i7 = n % 10;
    stack <long long> win_kombo;
    long long temp;
    if(mas[i1][i2][i3][i4][i5][i6][i7] == 0)
    {
        fout << "The first wins" << endl;
        for(int k = 1; k < 4; k++)
        {
            if(mas[i1+k][i2][i3][i4][i5][i6][i7] == 1)
            {
                temp = i7 + i6*10 + i5*100 + i4*1000 + i3*10000 + i2*100000 + (i1+k)*1000000;
                win_kombo.push(temp);
            }
        }
        for(int k = 1; k < 4; k++)
        {
             if(mas[i1][i2+k][i3][i4][i5][i6][i7] == 1)
            {
                temp = i7 + i6*10 + i5*100 + i4*1000 + i3*10000 + (i2+k)*100000 + i1*1000000;
                win_kombo.push(temp);                
            }           
        }
        for(int k = 1; k < 4; k++)
        {
             if(mas[i1][i2][i3+k][i4][i5][i6][i7] == 1)
            {
                temp = i7 + i6*10 + i5*100 + i4*1000 + (i3+k)*10000 + i2*100000 + i1*1000000;
                win_kombo.push(temp);                
            }           
        }
        for(int k = 1; k < 4; k++)
        {
            if(mas[i1][i2][i3][i4+k][i5][i6][i7] == 1)
            {
                temp = i7 + i6*10 + i5*100 + (i4+k)*1000 + i3*10000 + i2*100000 + i1*1000000;
                win_kombo.push(temp);               
            }           
        }
        for(int k = 1; k < 4; k++)
        {
            if(mas[i1][i2][i3][i4][i5+k][i6][i7] == 1)
            {
                temp = i7 + i6*10 + (i5+k)*100 + i4*1000 + i3*10000 + i2*100000 + i1*1000000;
                win_kombo.push(temp);
            }            
        }
        for(int k = 1; k < 4; k++)
        {
             if(mas[i1][i2][i3][i4][i5][i6+k][i7] == 1)
            {
                temp = i7 + (i6+k)*10 + i5*100 + i4*1000 + i3*10000 + i2*100000 + i1*1000000;
                win_kombo.push(temp);              
            }           
        }
        for(int k = 1; k < 4; k++)
        {
            if(mas[i1][i2][i3][i4][i5][i6][i7+k] == 1)
            {
                temp = (i7+k) + i6*10 + i5*100 + i4*1000 + i3*10000 + i2*100000 + i1*1000000;
                win_kombo.push(temp);                
            }            
        }
        while(!win_kombo.empty())
        {
            fout << win_kombo.top() << " ";
            win_kombo.pop();
        }
    }
    else
    {
        fout << "The second wins" << endl;
    }
}