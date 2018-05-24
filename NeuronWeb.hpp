#ifndef NEURON_WEB_HPP
#define NEURON_WEB_HPP

#include <array>
#include <random>

#include "Matrix.hpp"

/*
 * Neuron layer
 */
template<typename T, size_t N>
using Layer = std::array<T, N>;


/*
 * Веса в нейронной сети. Являются матрицей. При создании весов
 * необходимо указать тип, которым будет представлены весы,
 * количество входных нейроннов+1 и количество выходных
 * нейроннов. +1 для входных нейронов необходим т.к. дополнительный
 * входной нейрон представляет собой смещения для выходных
 * нейроннов.
 */
template<typename T, size_t INC_plus_one, size_t OUTC>
using Weights = clever::Matrix<T, INC_plus_one, OUTC>;


/*
 * initialize weights by random values
 */
template<typename T, size_t INC_plus_one, size_t OUTC>
void init_weights_random(Weights<T, INC_plus_one, OUTC> &w)
{
	for(size_t o = 0; o < OUTC; ++o) {
		for(size_t i = 0; i < INC_plus_one; ++i) {
			w[i][o] = double(rand())/RAND_MAX;
		}
	}
	return;
}



/*
 * Алгоритм прямого распространения
 */
template<typename T, size_t INC, size_t OUTC>
void forward_propagation(
	Layer<T, INC> const &in,
	Layer<T, OUTC> &out,
	Weights<T, INC+1, OUTC> const &w
)
{
#define sigmoid(x) ( 1.0f / ( 1.0f + exp(-(x)) ) )
	for(size_t o = 0; o < OUTC; ++o) {
		out[o] = 0;
		for(size_t i = 0; i < INC; ++i) {
			out[o] += in[i]*w[i][o];
		}
		out[o] += w[INC][o];
		out[o] = sigmoid(out[o]);
	}
	return;
}



/*
 * Узнать ошибки по эталону
 */
template<typename T, size_t OUTC>
void learn_errors(
	Layer<T, OUTC> const &out,
	std::array<T, OUTC> const &ideal,
	std::array<T, OUTC> &err
)
{
	for(size_t i = 0; i < OUTC; ++i) {
		err[i] = (ideal[i] - out[i]) * out[i] * (1 - out[i]);
	}
	return;
}



/*
 * Узнать ошибки по ошибкам из верхнего слоя
 */
template<typename T, size_t INC, size_t OUTC>
void learn_errors(
	Layer<T, INC> const &in,
	Weights<T, INC+1, OUTC> const &w,
	std::array<T, OUTC> const &erro,
	std::array<T, INC> &erri
)
{
	for(size_t i = 0; i < INC; ++i) {
		erri[i] = 0;
		for(size_t o = 0; o < OUTC; ++o) {
			erri[i] += erro[o] * w[i][o];
		}
		erri[i] *= in[i] * (1 - in[i]);
	}
	return;
}



/*
 * Изменение весов - обратное распространение
 */
template<typename T, std::size_t OUTC, std::size_t INC>
void reverse_propagation(
	Layer<T, OUTC> const &out,
	Layer<T, INC> const &in,
	Weights<T, INC+1, OUTC> &w,
	std::array<T, OUTC> err
)
{
#define LEARN_RATE 0.2f
	for(size_t o = 0; o < OUTC; ++o) {
		for(size_t i = 0; i < INC; ++i) {
			w[i][o] += LEARN_RATE * err[o] * in[i];
		}
		w[INC][o] += LEARN_RATE * err[o];
	}
	return;
}



#endif // NEURON_WEB_HPP
