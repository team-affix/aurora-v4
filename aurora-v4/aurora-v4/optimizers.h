#pragma once
#include "affix-base/pch.h"

namespace aurora
{
	class optimizer
	{
	public:
		std::vector<state_gradient_pair*> m_values;

	public:
		optimizer(
			std::vector<state_gradient_pair*> a_values
		) :
			m_values(a_values)
		{

		}

		void normalize_gradients(

		)
		{
			double l_max_gradient_magnitude = 0;

			for (const auto& l_value : m_values)
			{
				double l_abs_gradient = std::abs(l_value->m_gradient);
				if (l_abs_gradient > l_max_gradient_magnitude)
					l_max_gradient_magnitude = l_abs_gradient;
			}

			for (auto& l_value : m_values)
			{
				l_value->m_gradient = l_value->m_gradient / l_max_gradient_magnitude;
			}

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
			std::vector<state_gradient_pair*> a_values,
			const double& a_learn_rate
		) :
			optimizer(a_values),
			m_learn_rate(a_learn_rate)
		{

		}

		virtual void update(

		)
		{
			normalize_gradients();
			for (auto& l_value : m_values)
			{
				l_value->m_state -= m_learn_rate * l_value->m_gradient;
				l_value->m_gradient = 0;
			}
		}

	};

	class gradient_descent_with_momentum : public optimizer
	{
	public:
		double m_learn_rate = 0;
		double m_beta = 0;
		double m_alpha = 0;
		std::vector<double> m_momenta;

	public:
		gradient_descent_with_momentum(
			std::vector<state_gradient_pair*> a_values,
			const double& a_learn_rate,
			const double& a_beta
		) :
			optimizer(a_values),
			m_learn_rate(a_learn_rate),
			m_beta(a_beta),
			m_alpha(1.0 - a_beta),
			m_momenta(a_values.size())
		{
			assert(a_beta >= 0 && a_beta <= 1);
		}

		virtual void update(

		)
		{
			normalize_gradients();
			for (int i = 0; i < m_values.size(); i++)
			{
				double& l_momentum = m_momenta[i];
				auto& l_value = m_values[i];
				l_momentum = m_beta * l_momentum + m_alpha * l_value->m_gradient;
				l_value->m_state -= m_learn_rate * l_momentum;
				l_value->m_gradient = 0;
			}
		}

	};

}
