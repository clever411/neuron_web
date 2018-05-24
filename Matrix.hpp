#ifndef CLEVER_MATRIX_HPP
#define CLEVER_MATRIX_HPP



namespace clever
{



/*
 * 	Класс. Представляет собой двумерную матрицу. Имеет
 * дополнительные функции для удобства. Сначала указывается
 * тип хранимый в матрице, затем количество строк и последним
 * количество столбцов. ЗАМЕТЬТЕ! сначала высота, потом ширина,
 * не наооборот. При доступе к матрице через оператор [a][b]
 * также как и при создании матрицы сначала указывается 
 * строка, затем столбец. СНАЧАЛА Y И ТОЛЬКО ПОТОМ X!!!
 * 	Структура является агрегатом. По умолчанию значения
 * неопределены.
 */

template<typename T, size_t H, size_t W>
struct Matrix
{
static_assert(W > 0);
static_assert(H > 0);
	
	typedef T value_type;
	typedef size_t size_type;

	constexpr static size_type const w = W;
	constexpr static size_type const h = H;
	constexpr static size_type const length = w*h;
	


	T d[h][w];



	value_type *operator[](int n)
	{
		return d[n];
	}
	value_type const *operator[](int n) const
	{
		return d[n];
	}



	/*
	 * Начало матрицы. В случае инкримента, указатель 
	 * переставляется на следующее в строке значение.
	 * Если  текущий элемент последний в строке и есть еще одна
	 * строка, то указатель установится на начало следующей строки.
	 * Если строк больше нет, указатель станет равен end().
	 */
	value_type *begin()
	{
		return (T*)d;
	}
	/*
	 * Конец матрицы
	 */
	value_type *end()
	{
		return (T*)d+w*h;
	}

	/*
	 * Константный версии функций, в лучших традициях STL. 
	 */
	value_type const *cend() const
	{
		return (T const *)d+w*h;
	}
	value_type const *cbegin() const
	{
		return (T const *)d;
	}



	/*
	 * Возвращает указатель установленный на начало указанной
	 * строки.
	 */
	value_type *begin(int line)
	{
		return d[line];
	}
	/*
	 * Возвращает указатель установленый в конец указанной
	 * строки.
	 * Будет удобно использовать для обработки строк с помощью
	 * алгоритмов STL.
	 */
	value_type *end(int line)
	{
		return d[line]+w;
	}

	value_type const *cbegin(int line) const
	{
		return d[line];
	}
	value_type const *cend(int line) const
	{
		return d[line]+w;
	}



	constexpr static size_type size()
	{
		return length;
	}

};



}



#endif // CLEVER_MATRIX_HPP
