#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"
#include "maths.h"

namespace aurora
{
	class element;

	struct model
	{
	private:
		static std::vector<model> s_models;
		static std::default_random_engine s_default_random_engine;
		static std::vector<state_gradient_pair>* m_parameters;
		static size_t m_parameter_index;

	private:
		std::vector<affix_base::data::ptr<element>> m_elements;

	private:
		model(

		)
		{

		}

	public:
		static void begin(

		);

		static model end(

		);

		static model end(
			const double& a_minimum_parameter_state,
			const double& a_maximum_parameter_state
		);

		static void insert(
			const affix_base::data::ptr<element>& a_element
		)
		{
			s_models.back().m_elements.push_back(a_element);
		}

	public:
		std::vector<state_gradient_pair>::iterator next_parameter(

		)
		{
			return m_next_parameter;
		}

		void fwd(

		);

		void bwd(

		);

		std::vector<affix_base::data::ptr<element>>& elements(

		)
		{
			return m_elements;
		}

		std::vector<state_gradient_pair>& parameters(

		)
		{
			return m_parameters;
		}

	};

	struct parameters
	{
	private:
		static std::vector<state_gradient_pair>* s_parameter_vector;
		static size_t s_next_index;

	public:
		static void bind(
			std::vector<state_gradient_pair>& a_parameter_vector
		)
		{
			s_parameter_vector = &a_parameter_vector;
			s_next_index = 0;
		}

		static void next_index(
			const size_t& a_next_index
		)
		{
			if (a_next_index > s_parameter_vector->size())
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
			if (s_next_index == s_parameter_vector->size())
			{
				s_parameter_vector->push_back(state_gradient_pair());
				return &s_parameter_vector->back();
			}
			else
			{
				return &s_parameter_vector->at(s_next_index);
			}
			throw std::exception("Error: reached end of parameters::next() function");
		}

	};

}
