#include <iostream>
#include <fstream>
#include <string>

using namespace std;


string const FILE_NAME = "test.in";
int const DEFAULT_MAX = 2;
int const DEFAULT_WIDTH = 4;

int maxc;
int *code;
int width;

void init_code()
{
	code = new int[width];
	for(int *b = code, *e = code+width; b != e; ++b) {
		*b = 0;
	}
	return;
}

bool next(int n)
{
	if(code[n] == maxc) {
		if(n == width-1)
			return false;
		code[n] = 0;
		return next(n+1);
	}
	else
		++code[n];

	return true;
}



int main(int argc, char const *argv[])
{
	try {
		maxc = argc > 1 ? stoi(argv[1]) : DEFAULT_MAX;
		if(maxc < 0)
			throw "invalid max";

		width = argc > 2 ? stoi(argv[2]) : DEFAULT_WIDTH;
		if(width <= 0)
			throw "invlaid width";

		init_code();

		do {
			for(int j = width-1; j >= 0; --j) {
				cout << code[j] << ' ';
			}
			cout << '\n';
		} while(next(0));

		return 0;
	}
	catch(invalid_argument const &) {
		cerr << "invalid argument";
	}
	catch(char const *m) {
		cerr << m << endl;
		return 1;
	}
}
