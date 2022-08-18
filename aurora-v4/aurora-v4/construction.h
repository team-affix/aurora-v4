#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"
#include "maths.h"

namespace aurora
{
	class element;

	struct element_vector : public std::vector<affix_base::data::ptr<element>>
	{
	private:
		static std::vector<element_vector> s_models;

	private:
		element_vector(

		)
		{

		}

	public:
		static void start(

		);

		static element_vector stop(

		);

		static void insert(
			const affix_base::data::ptr<element>& a_element
		)
		{
			s_models.back().push_back(a_element);
		}

	public:
		void fwd(

		);

		void bwd(

		);

	};

	struct parameter_vector : public std::vector<affix_base::data::ptr<state_gradient_pair>>
	{
	private:
		static parameter_vector s_parameter_vector;
		static size_t s_next_index;
		static std::default_random_engine s_default_random_engine;

	private:
		parameter_vector(

		)
		{

		}

	public:
		static void start(

		)
		{
			s_next_index = 0;
		}

		static parameter_vector stop(

		)
		{
			parameter_vector l_result = s_parameter_vector;
			s_parameter_vector = {};
			return l_result;
		}

		static parameter_vector stop(
			const double& a_parameter_minimum_value,
			const double& a_parameter_maximum_value
		)
		{
			parameter_vector l_result = stop();

			std::uniform_real_distribution<double> l_urd(a_parameter_minimum_value, a_parameter_maximum_value);
			for (auto& l_parameter : l_result)
				l_parameter->m_state = l_urd(s_default_random_engine);

			return l_result;

		}

		static void next_index(
			const size_t& a_next_index
		)
		{
			if (a_next_index > s_parameter_vector.size())
				throw std::exception("Error: a_parameter_index was out of legal bounds given s_parameter_vector's size.");
			s_next_index = a_next_index;
		}

		static size_t next_index(

		)
		{
			return s_next_index;
		}

		static state_gradient_pair* next(

		)
		{
			if (s_next_index == s_parameter_vector.size())
			{
				s_parameter_vector.push_back(new state_gradient_pair());
				s_next_index++;
				return s_parameter_vector.back();
			}
			else
			{
				affix_base::data::ptr<state_gradient_pair> l_result = s_parameter_vector[s_next_index];
				s_next_index++;
				return l_result;
			}
		}

	public:
		operator std::vector<state_gradient_pair*>(

		)
		{
			std::vector<state_gradient_pair*> l_result(size());

			for (int i = 0; i < size(); i++)
				l_result[i] = at(i);

			return l_result;

		}

	};

}
