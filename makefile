all: debug


debug: main.cpp
	g++ -o neuweb -g3 -Wall main.cpp

release: main.cpp
	g++ -o neuweb -O3 main.cpp

fill: fill.out
	fill.out

fill.out: fill_test_file.cpp
	g++ -o fill.out -g3 -Wall fill_test_file.cpp
