#ifndef NEURON_WEB_HPP
#define NEURON_WEB_HPP

#include <random>


/*
 * Класс представляет из себя слой нейронов.
 * Имеет вспомогательные функции.
 */
template<int C, typename ValueType = double>
struct NeuronLayer
{
	typedef ValueType value_type;

	constexpr static int const COUNT = C;

	value_type d[COUNT];

	NeuronLayer &init()
	{
		return init(value_type());
	}

	NeuronLayer &init(value_type defv)
	{
		for(int i = 0; i < COUNT; ++i) {
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
template<int INPUT_COUNT, int OUTPUT_COUNT, typename ValueType = double>
struct NeuronWeights
{
	typedef ValueType value_type;

	constexpr static int const INPUTC = INPUT_COUNT;
	constexpr static int const OUTPUTC = OUTPUT_COUNT;

	value_type d[INPUTC+1][OUTPUTC];

	NeuronWeights &init()
	{
		return init(value_type());
	}
	NeuronWeights &init(value_type defv = value_type())
	{
		for(int i = 0; i < INPUTC+1; ++i)
			for(int o = 0; o < OUTPUTC; ++o)
				d[i][o] = defv;
		return *this;
	}

	NeuronWeights &randomInit()
	{
		for(int i = 0; i < INPUTC+1; ++i)
			for(int o = 0; o < OUTPUTC; ++o)
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
template<int INC, int OUTC, typename ValueType>
void forward_propagation(
	NeuronLayer<INC, ValueType> const &in,
	NeuronLayer<OUTC, ValueType> &out,
	NeuronWeights<INC, OUTC, ValueType> const &weights
)
{
#define sigmoid(x) ( 1.0f / ( 1.0f + exp(-(x)) ) )
	for(int o = 0; o < OUTC; ++o) {
		out[o] = 0;
		for(int i = 0; i < INC; ++i) {
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
template<int OUTC, typename ValueType>
void learn_errors(
	NeuronLayer<OUTC> const &out,
	ValueType const ideal[OUTC],
	ValueType err[OUTC]
)
{
	for(int i = 0; i < OUTC; ++i) {
		err[i] = (ideal[i] - out[i]) * out[i] * (1 - out[i]);
	}
	return;
}



/*
 * Узнать ошибки по ошибкам из верхнего слоя
 */
template<int INC, int OUTC, typename ValueType>
void learn_errors(
	NeuronLayer<INC> const &in,
	NeuronWeights<INC, OUTC> const &w,
	ValueType const erro[OUTC],
	ValueType erri[INC]
)
{
	for(int i = 0; i < INC; ++i) {
		erri[i] = 0;
		for(int o = 0; o < OUTC; ++o) {
			erri[i] += erro[o] * w[i][o];
		}
		erri[i] *= in[i] * (1 - in[i]);
	}
	return;
}



/*
 * Изменение весов - обратное распространение
 */
template<int OUTC, int INC, typename ValueType>
void reverse_propagation(
	NeuronLayer<OUTC, ValueType> const &out,
	NeuronLayer<INC, ValueType> const &in,
	NeuronWeights<INC, OUTC, ValueType> &w,
	ValueType const err[OUTC]
)
{
#define LEARN_RATE 0.2f
	for(int o = 0; o < OUTC; ++o) {
		for(int i = 0; i < INC; ++i) {
			w[i][o] += LEARN_RATE * err[o] * in[i];
		}
		w[INC][o] += LEARN_RATE * err[o];
	}
	return;
}



#endif // NEURON_WEB_HPP
