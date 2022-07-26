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

	struct state_gradient_pair_vector : public std::vector<state_gradient_pair>
	{
		state_gradient_pair_vector(

		)
		{

		}

		state_gradient_pair_vector(
			const size_t& a_size
		)
		{
			resize(a_size);
		}

		state_gradient_pair_vector(
			const std::vector<state_gradient_pair>& a_values
		) :
			std::vector<state_gradient_pair>(a_values)
		{

		}

		state_gradient_pair_vector(
			const std::initializer_list<state_gradient_pair>& a_values
		) :
			std::vector<state_gradient_pair>(a_values)
		{

		}

		void set_state(
			const state_gradient_pair_vector& a_other
		)
		{
			assert(size() == a_other.size());
			for (int i = 0; i < size(); i++)
				at(i).m_state = a_other[i].m_state;
		}

		void clear_state(

		)
		{
			for (int i = 0; i < size(); i++)
				at(i).m_state = 0;
		}

		std::vector<state_gradient_pair*> pointers(

		)
		{
			std::vector<state_gradient_pair*> l_result;
			for (int i = 0; i < size(); i++)
				l_result.push_back(&at(i));
			return l_result;
		}

	};

	struct state_gradient_pair_matrix : public std::vector<state_gradient_pair_vector>
	{
		state_gradient_pair_matrix(

		)
		{

		}

		state_gradient_pair_matrix(
			const size_t& a_rows,
			const size_t& a_cols
		)
		{
			resize(a_rows);
			for (int i = 0; i < a_rows; i++)
				at(i).resize(a_cols);
		}

		state_gradient_pair_matrix(
			const std::vector<state_gradient_pair_vector>& a_values
		) :
			std::vector<state_gradient_pair_vector>(a_values)
		{

		}

		state_gradient_pair_matrix(
			const std::initializer_list<state_gradient_pair_vector>& a_values
		) :
			std::vector<state_gradient_pair_vector>(a_values)
		{

		}

		void set_state(
			const state_gradient_pair_matrix& a_other
		)
		{
			assert(size() == a_other.size());
			for (int i = 0; i < size(); i++)
				at(i).set_state(a_other[i]);
		}

		void clear_state(

		)
		{
			for (int i = 0; i < size(); i++)
				at(i).clear_state();
		}

		std::vector<std::vector<state_gradient_pair*>> pointers(

		)
		{
			std::vector<std::vector<state_gradient_pair*>> l_result;
			for (int i = 0; i < size(); i++)
			{
				l_result.push_back(at(i).pointers());
			}
			return l_result;
		}

	};

	class optimizer
	{
	public:
		state_gradient_pair* m_value = nullptr;

	public:
		optimizer(
			state_gradient_pair* a_value
		) :
			m_value(a_value)
		{

		}

		virtual void update(

		)
		{

		}

	};

	class gradient_descent : public optimizer
	{
	public:
		double m_learn_rate = 0;

	public:
		gradient_descent(
			state_gradient_pair* a_value,
			const double& a_learn_rate
		) :
			optimizer(a_value),
			m_learn_rate(a_learn_rate)
		{

		}

		virtual void update(

		)
		{
			m_value->m_state -= m_learn_rate * m_value->m_gradient;
			m_value->m_gradient = 0;
		}

	};
	
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

}
