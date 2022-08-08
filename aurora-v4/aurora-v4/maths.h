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

	inline double mean_squared_error(
		std::vector<state_gradient_pair*> a_predicted,
		const std::vector<state_gradient_pair>& a_desired
	)
	{
		assert(a_predicted.size() == a_desired.size());

		double l_result = 0;
		
		double l_coefficient = 1.0 / (double)a_predicted.size();

		double l_2_coefficient = 2.0 * l_coefficient;

		for (int i = 0; i < a_predicted.size(); i++)
		{
			state_gradient_pair& l_prediction = *a_predicted[i];
			const state_gradient_pair& l_desired = a_desired[i];
			double l_error = l_prediction.m_state - l_desired.m_state;
			double l_squared_error = l_error * l_error;
			l_result += l_coefficient * l_squared_error;
			l_prediction.m_gradient += l_2_coefficient * l_error;
		}

		return l_result;

	}

	inline double mean_squared_error(
		std::vector<std::vector<state_gradient_pair*>> a_predicted,
		const std::vector<std::vector<state_gradient_pair>>& a_desired
	)
	{
		assert(a_predicted.size() == a_desired.size());
		assert(a_predicted[0].size() == a_desired[0].size());

		double l_result = 0;

		double l_coefficient = 1.0 / (double)a_predicted.size() / (double)a_predicted[0].size();

		double l_2_coefficient = 2.0 * l_coefficient;

		for (int i = 0; i < a_predicted.size(); i++)
		{
			for (int j = 0; j < a_predicted[i].size(); j++)
			{
				state_gradient_pair& l_prediction = *a_predicted[i][j];
				const state_gradient_pair& l_desired = a_desired[i][j];
				double l_error = l_prediction.m_state - l_desired.m_state;
				double l_squared_error = l_error * l_error;
				l_result += l_coefficient * l_squared_error;
				l_prediction.m_gradient += l_2_coefficient * l_error;
			}
		}

		return l_result;

	}

	inline double cross_entropy(
		std::vector<state_gradient_pair*> a_predicted,
		const std::vector<state_gradient_pair>& a_desired
	)
	{
		return 0;
	}

	inline double cross_entropy(
		std::vector<std::vector<state_gradient_pair*>> a_predicted,
		const std::vector<std::vector<state_gradient_pair>>& a_desired
	)
	{
		return 0;
	}

}
