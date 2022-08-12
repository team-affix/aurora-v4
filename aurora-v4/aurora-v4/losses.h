#pragma once
#include "affix-base/pch.h"
#include "maths.h"

namespace aurora
{
	inline double mean_squared_error(
		state_gradient_pair* a_predicted,
		const state_gradient_pair& a_desired
	)
	{
		double l_error = a_predicted->m_state - a_desired.m_state;
		double l_result = l_error * l_error;
		a_predicted->m_gradient += 2.0 * l_error;
		return l_result;
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
