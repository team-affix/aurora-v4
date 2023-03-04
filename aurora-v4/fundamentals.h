#ifndef FUNDAMENTALS_H
#define FUNDAMENTALS_H

#include <vector>
#include <random>
#include <string>

namespace aurora
{
	inline std::default_random_engine i_default_random_engine(29);

	inline std::string to_string(
		const std::vector<double>& a_vector
	)
	{
		std::string l_result;
		for (const auto& l_value : a_vector)
			l_result += std::to_string(l_value) + " ";
		return l_result.substr(0, l_result.size() - 1);
	}

	inline std::string to_string(
		const std::vector<std::vector<double>>& a_matrix
	)
	{
		std::string l_result;
		for (const auto& l_vector : a_matrix)
			l_result += to_string(l_vector) + "\n";
		return l_result.substr(0, l_result.size() - 1);
	}

	inline std::string to_string(
		const std::vector<std::vector<std::vector<double>>>& a_tensor
	)
	{
		std::string l_result;
		for (int i = 0; i < a_tensor.size(); i++)
			l_result += "____(MATRIX " + std::to_string(i) + ")____\n" + to_string(a_tensor[i]) + "\n";
		return l_result.substr(0, l_result.size() - 1);
	}

}

#endif
