#ifndef FUNDAMENTALS_H
#define FUNDAMENTALS_H

#include <vector>
#include <random>
#include <string>
#include <ostream>
#include <initializer_list>
#include <array>

namespace aurora
{
    /// Defining some typedefs for improving the readability of code
    /// and user-friendliness.

    // We want a multidimensional array, the dimensions of which are defined at compile time.
    
    template<typename T, size_t I, size_t ... J>
    struct tensor : public std::array<tensor<T, J ...>, I>
    {

    };

    template<typename T, size_t I>
    struct tensor<T, I> : public std::array<T, I>
    {
        tensor(

        )
        {
            std::fill_n(std::array<T, I>::begin(), std::array<T, I>::size(), T());
        }
    };


    typedef std::vector<double>       state_vector;
    typedef std::vector<state_vector> state_matrix;
    typedef std::vector<state_matrix> state_cuboid;

	inline std::default_random_engine i_default_random_engine(29);

	inline std::string to_string(
		const state_vector& a_vector
	)
	{
		std::string l_result;
		for (const auto& l_value : a_vector)
			l_result += std::to_string(l_value) + " ";
		return l_result.substr(0, l_result.size() - 1);
	}

	inline std::string to_string(
		const state_matrix& a_matrix
	)
	{
		std::string l_result;
		for (const auto& l_vector : a_matrix)
			l_result += to_string(l_vector) + "\n";
		return l_result.substr(0, l_result.size() - 1);
	}

	inline std::string to_string(
		const state_cuboid& a_tensor
	)
	{
		std::string l_result;
		for (int i = 0; i < a_tensor.size(); i++)
			l_result += "____(MATRIX " + std::to_string(i) + ")____\n" + to_string(a_tensor[i]) + "\n";
		return l_result.substr(0, l_result.size() - 1);
	}

}

template<typename T, size_t I>
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element << ' ';

    return a_ostream;

}

template<typename T, size_t I, size_t J>
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I, J>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element << '\n';

    a_ostream << '\n';

    return a_ostream;

}

template<typename T, size_t I, size_t ... J>
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I, J ...>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element;

    return a_ostream;

}

template<typename T, size_t I, size_t ... J>
std::istream& operator>>(std::istream& a_istream, aurora::tensor<T, I, J ...>& a_tensor)
{
    for (auto& l_element : a_tensor)
        a_istream >> l_element;

    return a_istream;
    
}

#endif
