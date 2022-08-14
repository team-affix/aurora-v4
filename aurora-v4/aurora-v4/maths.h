#pragma once
#include "affix-base/pch.h"
#include <vector>

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

	inline std::vector<state_gradient_pair> vector(
		const size_t& a_size
	)
	{
		return std::vector<state_gradient_pair>(a_size);
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

	inline void set_state(
		state_gradient_pair& a_destination,
		const state_gradient_pair& a_source
	)
	{
		a_destination.m_state = a_source.m_state;
	}

	inline void set_state(
		std::vector<state_gradient_pair>& a_destination,
		const std::vector<state_gradient_pair>& a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			a_destination[i].m_state = a_source[i].m_state;
	}

	inline void set_state(
		std::vector<std::vector<state_gradient_pair>>& a_destination,
		const std::vector<std::vector<state_gradient_pair>>& a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			set_state(a_destination[i], a_source[i]);
	}

}
