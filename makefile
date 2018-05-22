all: main.out


main.out: main.cpp
	g++ -o main.out -g -Wall main.cpp

fill: fill.out
	fill.out

fill.out: fill_test_file.cpp
	g++ -o fill.out -g3 -Wall fill_test_file.cpp
