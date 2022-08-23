#pragma once
#include "affix-base/pch.h"
#include <vector>
#include "randomization.h"

namespace aurora
{
	struct state_gradient_pair
	{
		double m_state = 0;
		double m_gradient = 0;

		state_gradient_pair(

		)
		{

		}

		state_gradient_pair(
			const double& a_state
		) :
			m_state(a_state)
		{

		}

	};

	inline std::vector<state_gradient_pair*> pointers(
		std::vector<state_gradient_pair>& a_vector
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_vector.size(); i++)
			l_result.push_back(a_vector.data() + i);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> pointers(
		std::vector<std::vector<state_gradient_pair>>& a_matrix
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		for (int i = 0; i < a_matrix.size(); i++)
			l_result.push_back(pointers(a_matrix[i]));
		return l_result;
	}

	inline std::vector<state_gradient_pair> get_state(
		std::vector<state_gradient_pair*> a_vector
	)
	{
		std::vector<state_gradient_pair> l_result;
		for (auto& l_value : a_vector)
			l_result.push_back(l_value->m_state);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair>> get_state(
		std::vector<std::vector<state_gradient_pair*>> a_matrix
	)
	{
		std::vector<std::vector<state_gradient_pair>> l_result;
		for (int i = 0; i < a_matrix.size(); i++)
			l_result.push_back(get_state(a_matrix[i]));
		return l_result;
	}

	inline void set_state(
		std::vector<state_gradient_pair*> a_destination,
		const std::vector<state_gradient_pair*> a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			a_destination[i]->m_state = a_source[i]->m_state;
	}

	inline void set_state(
		std::vector<std::vector<state_gradient_pair*>> a_destination,
		const std::vector<std::vector<state_gradient_pair*>> a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			set_state(a_destination[i], a_source[i]);
	}

	inline void add_gradient(
		std::vector<state_gradient_pair*> a_destination,
		std::vector<state_gradient_pair*> a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			a_destination[i]->m_gradient += a_source[i]->m_gradient;
	}

	inline void add_gradient(
		std::vector<std::vector<state_gradient_pair*>> a_destination,
		std::vector<std::vector<state_gradient_pair*>> a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			add_gradient(a_destination[i], a_source[i]);
	}

	inline void clear_gradient(
		std::vector<state_gradient_pair*> a_vector
	)
	{
		for (auto& l_value : a_vector)
			l_value->m_gradient = 0;
	}

	inline void clear_gradient(
		std::vector<std::vector<state_gradient_pair*>> a_matrix
	)
	{
		for (auto& l_vector : a_matrix)
			clear_gradient(l_vector);
	}

	inline void randomize(
		std::vector<state_gradient_pair*> a_x,
		const double& a_minimum_random_value,
		const double& a_maximum_random_value
	)
	{
		std::uniform_real_distribution<double> l_urd(a_minimum_random_value, a_maximum_random_value);
		for (auto& l_value : a_x)
		{
			l_value->m_state = l_urd(i_default_random_engine);
		}
	}

	inline void randomize(
		std::vector<std::vector<state_gradient_pair*>> a_x,
		const double& a_minimum_random_value,
		const double& a_maximum_random_value
	)
	{
		for (auto& l_value : a_x)
		{
			randomize(l_value, a_minimum_random_value, a_maximum_random_value);
		}
	}

	inline std::vector<state_gradient_pair> vector(
		const size_t& a_size
	)
	{
		return std::vector<state_gradient_pair>(a_size);
	}

	inline std::vector<state_gradient_pair> vector(
		const size_t& a_size,
		const double& a_minimum_random_value,
		const double& a_maximum_random_value
	)
	{
		auto l_result = vector(a_size);
		randomize(pointers(l_result), a_minimum_random_value, a_maximum_random_value);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair>> matrix(
		const size_t& a_rows,
		const size_t& a_cols
	)
	{
		std::vector<std::vector<state_gradient_pair>> l_result;
		for (int i = 0; i < a_rows; i++)
			l_result.push_back(std::vector<state_gradient_pair>(a_cols));
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair>> matrix(
		const size_t& a_rows,
		const size_t& a_cols,
		const double& a_minimum_random_value,
		const double& a_maximum_random_value
	)
	{
		auto l_result = matrix(a_rows, a_cols);
		randomize(pointers(l_result), a_minimum_random_value, a_maximum_random_value);
		return l_result;
	}

}
