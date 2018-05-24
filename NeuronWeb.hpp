#ifndef NEURON_WEB_HPP
#define NEURON_WEB_HPP

#include <array>
#include <random>


// Neuron layer
template<typename T, size_t N>
using Layer = std::array<T, N>;



/*
 * Структура представляет из себя матрицу весов, имеет вспомогательные
 * функции.
 */
template<size_t INPUT_COUNT, size_t OUTPUT_COUNT, typename ValueType = double>
struct NeuronWeights
{
	typedef ValueType value_type;

	constexpr static size_t const INPUTC = INPUT_COUNT;
	constexpr static size_t const OUTPUTC = OUTPUT_COUNT;

	value_type d[INPUTC+1][OUTPUTC];

	NeuronWeights &init()
	{
		return init(value_type());
	}
	NeuronWeights &init(value_type defv = value_type())
	{
		for(std::size_t i = 0; i < INPUTC+1; ++i)
			for(std::size_t o = 0; o < OUTPUTC; ++o)
				d[i][o] = defv;
		return *this;
	}

	NeuronWeights &randomInit()
	{
		for(size_t i = 0; i < INPUTC+1; ++i)
			for(size_t o = 0; o < OUTPUTC; ++o)
				d[i][o] = double(rand())/RAND_MAX;
		return *this;
	}


	value_type const *operator[](int i) const
	{
		return d[i];
	}
	value_type *operator[](int i)
	{
		return d[i];
	}

};




/*
 * Алгоритм прямого распространения
 */
template<typename T, size_t INC, size_t OUTC>
void forward_propagation(
	Layer<T, INC> const &in,
	Layer<T, OUTC> &out,
	NeuronWeights<INC, OUTC, T> const &weights
)
{
#define sigmoid(x) ( 1.0f / ( 1.0f + exp(-(x)) ) )
	for(size_t o = 0; o < OUTC; ++o) {
		out[o] = 0;
		for(size_t i = 0; i < INC; ++i) {
			out[o] += in[i]*weights[i][o];
		}
		out[o] += weights[INC][o];
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
	NeuronWeights<INC, OUTC, T> const &w,
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
	NeuronWeights<INC, OUTC, T> &w,
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
