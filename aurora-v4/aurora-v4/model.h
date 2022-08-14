#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"
#include "maths.h"
#include "optimizers.h"

namespace aurora
{
	class element;

	struct model
	{
	private:
		static std::vector<model> s_models;
		static std::default_random_engine s_default_random_engine;

	private:
		std::vector<affix_base::data::ptr<element>> m_elements;
		std::vector<affix_base::data::ptr<state_gradient_pair>> m_parameters;
		std::vector<affix_base::data::ptr<optimizer>> m_optimizers;

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

		static model end(
			const double& a_minimum_parameter_state,
			const double& a_maximum_parameter_state,
			const std::function<affix_base::data::ptr<optimizer>(affix_base::data::ptr<state_gradient_pair>)>& a_construct_optimizer
		);

		static void insert(
			const affix_base::data::ptr<element>& a_element
		)
		{
			s_models.back().m_elements.push_back(a_element);
		}

		static void insert(
			const affix_base::data::ptr<state_gradient_pair>& a_parameter
		)
		{
			s_models.back().m_parameters.push_back(a_parameter);
		}

		static void insert(
			const std::vector<affix_base::data::ptr<element>>& a_elements
		)
		{
			s_models.back().m_elements.insert(s_models.back().m_elements.end(), a_elements.begin(), a_elements.end());
		}

		static void insert(
			const std::vector<affix_base::data::ptr<state_gradient_pair>>& a_parameters
		)
		{
			s_models.back().m_parameters.insert(s_models.back().m_parameters.end(), a_parameters.begin(), a_parameters.end());
		}

	public:
		void fwd(

		);

		void bwd(

		);

		void update(

		);

		std::vector<affix_base::data::ptr<element>>& elements(

		)
		{
			return m_elements;
		}

		std::vector<affix_base::data::ptr<state_gradient_pair>>& parameters(

		)
		{
			return m_parameters;
		}

	};

}
