#include <stdio.h>
#include <stdlib.h>

void main()
{
	FILE* file_in=fopen("test.csv","r");
	FILE* file_out=fopen("test_f.csv","w+");
	char c[2];
	unsigned long i=0;
	do
	{
		c[i%2]=fgetc(file_in);
		if(feof(file_in))
			break;
		if(i>1 && (c[0]==',' && c[1]==','))
			fprintf(file_out,"-1");
		if(i>1 && (c[0]==',' && c[1]=='\n'))
			fprintf(file_out,"-1");
		if(i>1 && (c[1]==',' && c[0]=='\n'))
			fprintf(file_out,"-1");
		if(c[i%2]>= 65 && c[i%2]<=90)
			fprintf(file_out,"%d",c[i%2]-65);
		else
			fprintf(file_out,"%c",c[i%2]);
		i++;
	}while(1);
}
