all: release


debug: main.cpp
	g++ -o neuweb -g3 -Wall main.cpp

release: main.cpp
	g++ -o neuweb -O3 main.cpp


fill: fill_test_file.cpp
	g++ -o fill -O3 fill_test_file.cpp
