#ifndef CLEVER_IOSTREAM_FUNCTIONS_HPP
#define CLEVER_IOSTREAM_FUNCTIONS_HPP

namespace clever
{
	template<class Ostream, class Matrix>
	Ostream &print_matrix(Ostream &os, Matrix const &m, int w, int h)
	{
		for(int i = 0; i < h; ++i) {
			for(int j = 0; j < w; ++j) {
				os << m[i][j] << '\t';
			}
			os << '\n';
		}
		return os;
	}
};

#endif // CLEVER_IOSTREAM_FUNCTIONS_HPP

