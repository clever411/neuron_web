#include <array>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <utility>

#include "NeuronWeb.hpp"

#include <clever/IostreamFunctions.hpp>

using namespace clever;
using namespace std;


typedef struct
{
	double health;
	double knife;
	double gun;
	double enemy;
} Input;

template<class Istream> Istream &operator>>(Istream &is, Input &input)
{
	is >> input.health;
	is >> input.knife;
	is >> input.gun;
	is >> input.enemy;
	return is;
}




typedef struct
{
	double attack;
	double run;
	double wander;
	double hide;

} Output;




constexpr pair<Input, Output> const samples[18] = 
{
/*	  health  knf   gun   enemy     attck  run   wand  hide	  */
/* 1  */{ { 2.0f, 0.0f, 0.0f, 0.0f  }, { 0.0f, 0.0f, 1.0f, 0.0f } },
/* 2  */{ { 2.0f, 1.0f, 1.0f, 0.0f  }, { 0.0f, 0.0f, 1.0f, 0.0f } },
/* 3  */{ { 2.0f, 0.0f, 1.0f, 1.0f  }, { 1.0f, 0.0f, 0.0f, 0.0f } },
/* 4  */{ { 2.0f, 0.0f, 1.0f, 2.0f  }, { 1.0f, 0.0f, 0.0f, 0.0f } },
/* 5  */{ { 2.0f, 1.0f, 0.0f, 2.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },
/* 6  */{ { 2.0f, 1.0f, 0.0f, 1.0f  }, { 1.0f, 0.0f, 0.0f, 0.0f } },

/* 7  */{ { 1.0f, 0.0f, 0.0f, 0.0f  }, { 0.0f, 0.0f, 1.0f, 0.0f } },
/* 8  */{ { 1.0f, 0.0f, 0.0f, 1.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },
/* 9  */{ { 1.0f, 0.0f, 1.0f, 1.0f  }, { 1.0f, 0.0f, 0.0f, 0.0f } },
/* 10 */{ { 1.0f, 0.0f, 1.0f, 2.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },
/* 11 */{ { 1.0f, 1.0f, 0.0f, 2.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },
/* 12 */{ { 1.0f, 1.0f, 0.0f, 1.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },

/* 13 */{ { 0.0f, 0.0f, 0.0f, 0.0f  }, { 0.0f, 0.0f, 1.0f, 0.0f } },
/* 14 */{ { 0.0f, 0.0f, 0.0f, 1.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },
/* 15 */{ { 0.0f, 0.0f, 1.0f, 1.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } },
/* 16 */{ { 0.0f, 0.0f, 1.0f, 2.0f  }, { 0.0f, 1.0f, 0.0f, 0.0f } },
/* 17 */{ { 0.0f, 1.0f, 0.0f, 2.0f  }, { 0.0f, 1.0f, 0.0f, 0.0f } },
/* 18 */{ { 0.0f, 1.0f, 0.0f, 1.0f  }, { 0.0f, 0.0f, 0.0f, 1.0f } }
};

	

string const DEFAULT_INPUT_TEST_FILE_NAME = "input.txt";
string const DEFAULT_OUTPUT_TEST_FILE_NAME = "output.txt";



#define INPUT_NEURONS 4
#ifndef HIDDEN_NEURONS
	#define HIDDEN_NEURONS 3
#endif
#define OUTPUT_NEURONS 4



int main(int argc, char const *argv[])
{
	// initialization
	srand(time(0));


	Layer<double, INPUT_NEURONS> input;
	Layer<double, HIDDEN_NEURONS> hidden;
	Layer<double, OUTPUT_NEURONS> output;


	NeuronWeights<INPUT_NEURONS, HIDDEN_NEURONS> ihw;
	NeuronWeights<HIDDEN_NEURONS, OUTPUT_NEURONS> how;

	ihw.randomInit();
	how.randomInit();

	// learning
	{

		ofstream info("info.txt");
		if(info.is_open())
			info << "[I] - Iteration, [S] - Sample, [E] - Error\n";
		else
			cerr << "can't open info.txt file" << endl;

		array<double, OUTPUT_NEURONS> ideal;
		array<double, OUTPUT_NEURONS> erro;
		array<double, HIDDEN_NEURONS> errh;
		double mse;

		cout << "learning..." << endl;
		for(int i = 0, s = 0; i < 100000ll; ++i, ++s) {
			if(s == 18)
				s = 0;

			input[0] = samples[s].first.health;
			input[1] = samples[s].first.knife;
			input[2] = samples[s].first.gun;
			input[3] = samples[s].first.enemy;
				
			ideal[0] = samples[s].second.attack;
			ideal[1] = samples[s].second.run;
			ideal[2] = samples[s].second.wander;
			ideal[3] = samples[s].second.hide;


			forward_propagation(input, hidden, ihw);
			forward_propagation(hidden, output, how);

			learn_errors(output, ideal, erro);
			learn_errors(hidden, how, erro, errh);

			reverse_propagation(output, hidden, how, erro);
			reverse_propagation(hidden, input, ihw, errh);

			if(info.is_open()) {
				mse = 0.0f;
				for(int i = 0; i < 4; ++i) {
					mse += erro[i]*erro[i];
				}
				mse /= 4;
				info << "[I] = " << i << ",\t[S] = " << s <<
					",\t[E] = " << mse << "\n";
			}
		}
		cout << "learning complete" << endl;
	}

	cout << "testing from file..." << endl;
	Input in;
	{
		string inputfilename = argc > 1 ?
			string(argv[1]) : DEFAULT_INPUT_TEST_FILE_NAME;
		ifstream testin(inputfilename);
		if(!testin.is_open()) {
			cerr << "can't open input test file '" <<
				inputfilename << "'" << endl;
			goto fail_testing_from_file_label;
		}
		
		string outputfilename = argc > 2 ?
			string(argv[2]) : DEFAULT_OUTPUT_TEST_FILE_NAME;
		ofstream testout(outputfilename);
		if(!testout.is_open()) {
			cerr << "can't open output test file" << outputfilename << endl;
			goto fail_testing_from_file_label;
		}
		

		// for beauty
		testout.setf(testout.fixed);
		testout << setprecision(3);

		while(testin >> in) {
			input[0] = in.health;
			input[1] = in.knife;
			input[2] = in.gun;
			input[3] = in.enemy;

			forward_propagation(input, hidden, ihw);
			forward_propagation(hidden, output, how);

			testout << "Input:   H: " << in.health <<
				", K: " << in.knife <<
				", G: " << in.gun <<
				", E: " << in.enemy << '\n';
			testout << "Attack:  " << output[0] << '\n';
			testout << "Run:     " << output[1] << '\n';
			testout << "Wander:  " << output[2] << '\n';
			testout << "Hide:    " << output[3] << '\n';
			testout << '\n';
		}
		goto success_testing_from_file_label;
	}
fail_testing_from_file_label:
		cout << "testing from file fail" << endl;
		goto end_testing_from_file_label;

success_testing_from_file_label:
		cout << "testing from file success" << endl;
		goto end_testing_from_file_label;

end_testing_from_file_label:


	// for beauty again
	cout.setf(cout.fixed);
	cout << setprecision(3);

	// testing form cin
	cout << "-------------------------------------------------" << endl;
	cout << "Start testing from character input" << endl;
	cout << "Enter parameters" << endl;
	while(true) {
		cout << "Health:  "; if(!(cin >> in.health)) break;
		cout << "Knife:   "; if(!(cin >> in.knife)) break;
		cout << "Gun:     "; if(!(cin >> in.gun)) break;
		cout << "Enemy:   "; if(!(cin >> in.enemy)) break;
		cout << endl;

		input[0] = in.health;
		input[1] = in.knife;
		input[2] = in.gun;
		input[3] = in.enemy;

		forward_propagation(input, hidden, ihw);
		forward_propagation(hidden, output, how);

		cout << "Action: " << endl;
			cout << "\tAttack:  " << output[0] << endl;
			cout << "\tRun:     " << output[1] << endl;
			cout << "\tWander:  " << output[2] << endl;
			cout << "\tHide:    " << output[3] << endl;
		
		cout << "-------------------------------------------------" << endl << endl;
	}


	return 0;
}
