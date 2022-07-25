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

}
