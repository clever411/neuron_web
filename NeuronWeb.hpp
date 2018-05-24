#ifndef NEURON_WEB_HPP
#define NEURON_WEB_HPP

#include <random>


/*
 * Класс представляет из себя слой нейронов.
 * Имеет вспомогательные функции.
 */
template<std::size_t C, typename ValueType = double>
struct NeuronLayer
{
	typedef ValueType value_type;

	constexpr static std::size_t const COUNT = C;

	value_type d[COUNT];

	NeuronLayer &init()
	{
		return init(value_type());
	}

	NeuronLayer &init(value_type defv)
	{
		for(std::size_t i = 0; i < COUNT; ++i) {
			d[i] = defv;
		}
		return *this;
	}

	/*
	 * Доступ
	 */
	value_type const &operator[](int i) const
	{
		return d[i];
	}

	value_type &operator[](int i)
	{
		return d[i];
	}
};



/*
 * Структура представляет из себя матрицу весов, имеет вспомогательные
 * функции.
 */
template<std::size_t INPUT_COUNT, std::size_t OUTPUT_COUNT, typename ValueType = double>
struct NeuronWeights
{
	typedef ValueType value_type;

	constexpr static std::size_t const INPUTC = INPUT_COUNT;
	constexpr static std::size_t const OUTPUTC = OUTPUT_COUNT;

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
		for(std::size_t i = 0; i < INPUTC+1; ++i)
			for(std::size_t o = 0; o < OUTPUTC; ++o)
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
template<std::size_t INC, std::size_t OUTC, typename ValueType>
void forward_propagation(
	NeuronLayer<INC, ValueType> const &in,
	NeuronLayer<OUTC, ValueType> &out,
	NeuronWeights<INC, OUTC, ValueType> const &weights
)
{
#define sigmoid(x) ( 1.0f / ( 1.0f + exp(-(x)) ) )
	for(std::size_t o = 0; o < OUTC; ++o) {
		out[o] = 0;
		for(std::size_t i = 0; i < INC; ++i) {
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
template<typename T, std::size_t OUTC>
void learn_errors(
	NeuronLayer<OUTC, T> const &out,
	std::array<T, OUTC> const &ideal,
	std::array<T, OUTC> &err
)
{
	for(std::size_t i = 0; i < OUTC; ++i) {
		err[i] = (ideal[i] - out[i]) * out[i] * (1 - out[i]);
	}
	return;
}



/*
 * Узнать ошибки по ошибкам из верхнего слоя
 */
template<typename T, std::size_t INC, std::size_t OUTC>
void learn_errors(
	NeuronLayer<INC, T> const &in,
	NeuronWeights<INC, OUTC, T> const &w,
	std::array<T, OUTC> const &erro,
	std::array<T, INC> &erri
)
{
	for(std::size_t i = 0; i < INC; ++i) {
		erri[i] = 0;
		for(std::size_t o = 0; o < OUTC; ++o) {
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
	NeuronLayer<OUTC, T> const &out,
	NeuronLayer<INC, T> const &in,
	NeuronWeights<INC, OUTC, T> &w,
	std::array<T, OUTC> err
)
{
#define LEARN_RATE 0.2f
	for(std::size_t o = 0; o < OUTC; ++o) {
		for(std::size_t i = 0; i < INC; ++i) {
			w[i][o] += LEARN_RATE * err[o] * in[i];
		}
		w[INC][o] += LEARN_RATE * err[o];
	}
	return;
}



#endif // NEURON_WEB_HPP
