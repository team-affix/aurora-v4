#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"
#include <vector>

namespace aurora
{
	inline std::default_random_engine i_default_random_engine(28);

	struct state_gradient_pair_dependency
	{
		double& m_state;
		double& m_partial_gradient;
	};

	struct state_gradient_pair
	{
		double m_state = 0;

		std::vector<affix_base::data::ptr<double>> m_partial_gradients;

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

		double gradient(

		)
		{
			double l_result = 0;
			for (auto& l_partial_gradient : m_partial_gradients)
			{
				l_result += *l_partial_gradient;
			}
			return l_result;
		}

		state_gradient_pair_dependency depend(

		)
		{
			m_partial_gradients.push_back(new double(0));
			return state_gradient_pair_dependency{ m_state, *m_partial_gradients.back() };
		}

	};

	inline std::vector<state_gradient_pair*> pointers(
		std::vector<state_gradient_pair>& a_vector
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_vector.size(); i++)
			l_result.push_back(&a_vector[i]);
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

	inline std::vector<std::vector<std::vector<state_gradient_pair*>>> pointers(
		std::vector<std::vector<std::vector<state_gradient_pair>>>& a_tensor
	)
	{
		std::vector<std::vector<std::vector<state_gradient_pair*>>> l_result;
		for (int i = 0; i < a_tensor.size(); i++)
			l_result.push_back(pointers(a_tensor[i]));
		return l_result;
	}

	inline std::vector<double> get_state(
		const std::vector<state_gradient_pair*>& a_vector
	)
	{
		std::vector<double> l_result(a_vector.size());
		for (int i = 0; i < a_vector.size(); i++)
			l_result[i] = a_vector[i]->m_state;
		return l_result;
	}

	inline std::vector<std::vector<double>> get_state(
		const std::vector<std::vector<state_gradient_pair*>>& a_matrix
	)
	{
		std::vector<std::vector<double>> l_result(a_matrix.size());
		for (int i = 0; i < a_matrix.size(); i++)
			l_result[i] = get_state(a_matrix[i]);
		return l_result;
	}

	inline std::vector<std::vector<std::vector<double>>> get_state(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_tensor
	)
	{
		std::vector<std::vector<std::vector<double>>> l_result(a_tensor.size());
		for (int i = 0; i < a_tensor.size(); i++)
			l_result[i] = get_state(a_tensor[i]);
		return l_result;
	}

	inline void set_state(
		const std::vector<state_gradient_pair*>& a_destination,
		const std::vector<double>& a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			a_destination[i]->m_state = a_source[i];
	}

	inline void set_state(
		const std::vector<std::vector<state_gradient_pair*>>& a_destination,
		const std::vector<std::vector<double>>& a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			set_state(a_destination[i], a_source[i]);
	}

	inline void set_state(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_destination,
		const std::vector<std::vector<std::vector<double>>>& a_source
	)
	{
		for (int i = 0; i < a_destination.size(); i++)
			set_state(a_destination[i], a_source[i]);
	}

	inline std::vector<double> get_gradient(
		const std::vector<state_gradient_pair*>& a_vector
	)
	{
		std::vector<double> l_result(a_vector.size());
		for (int i = 0; i < a_vector.size(); i++)
			l_result[i] = a_vector[i]->gradient();
		return l_result;
	}

	inline std::vector<std::vector<double>> get_gradient(
		const std::vector<std::vector<state_gradient_pair*>>& a_matrix
	)
	{
		std::vector<std::vector<double>> l_result(a_matrix.size());
		for (int i = 0; i < a_matrix.size(); i++)
			l_result[i] = get_state(a_matrix[i]);
		return l_result;
	}

	inline std::vector<std::vector<std::vector<double>>> get_gradient(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_tensor
	)
	{
		std::vector<std::vector<std::vector<double>>> l_result(a_tensor.size());
		for (int i = 0; i < a_tensor.size(); i++)
			l_result[i] = get_state(a_tensor[i]);
		return l_result;
	}

	inline void randomize_state(
		const std::vector<state_gradient_pair*>& a_x,
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

	inline void randomize_state(
		const std::vector<std::vector<state_gradient_pair*>>& a_x,
		const double& a_minimum_random_value,
		const double& a_maximum_random_value
	)
	{
		for (auto& l_value : a_x)
		{
			randomize_state(l_value, a_minimum_random_value, a_maximum_random_value);
		}
	}

	inline void randomize_state(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_x,
		const double& a_minimum_random_value,
		const double& a_maximum_random_value
	)
	{
		for (auto& l_value : a_x)
		{
			randomize_state(l_value, a_minimum_random_value, a_maximum_random_value);
		}
	}

	inline std::vector<state_gradient_pair> input(
		const size_t& a_size
	)
	{
		return std::vector<state_gradient_pair>(a_size);
	}

	inline std::vector<std::vector<state_gradient_pair>> input(
		const size_t& a_rows,
		const size_t& a_cols
	)
	{
		std::vector<std::vector<state_gradient_pair>> l_result(a_rows);
		for (int i = 0; i < a_rows; i++)
			l_result[i] = input(a_cols);
		return l_result;
	}
	
	inline std::vector<std::vector<std::vector<state_gradient_pair>>> input(
		const size_t& a_depth,
		const size_t& a_rows,
		const size_t& a_cols
	)
	{
		std::vector<std::vector<std::vector<state_gradient_pair>>> l_result(a_depth);
		for (int i = 0; i < a_depth; i++)
			l_result[i] = input(a_rows, a_cols);
		return l_result;
	}

}
