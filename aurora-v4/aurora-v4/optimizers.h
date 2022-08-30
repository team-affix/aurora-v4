#pragma once
#include "affix-base/pch.h"
#include "maths.h"

namespace aurora
{
	class optimizer
	{
	public:
		std::vector<state_gradient_pair*> m_values;

	public:
		optimizer(
			const std::vector<state_gradient_pair*>& a_values
		) :
			m_values(a_values.begin(), a_values.end())
		{

		}

		virtual void update(

		)
		{

		}

	protected:
		std::vector<double> normalized_gradients(

		)
		{
			double l_normalization_denominator = 0;

			std::vector<double> l_gradients = get_gradient(m_values);

			for (auto& l_gradient : l_gradients)
				l_normalization_denominator += std::abs(l_gradient);

			for (auto& l_gradient : l_gradients)
				l_gradient /= l_normalization_denominator;

			return l_gradients;

		}

	};

	class gradient_descent : public optimizer
	{
	public:
		double m_learn_rate = 0;

	public:
		gradient_descent(
			const std::vector<state_gradient_pair*>& a_values,
			const double& a_learn_rate
		) :
			optimizer(a_values),
			m_learn_rate(a_learn_rate)
		{

		}

		virtual void update(

		)
		{
			std::vector<double> l_gradients = normalized_gradients();
			for (int i = 0; i < m_values.size(); i++)
			{
				m_values[i]->m_state -= m_learn_rate * l_gradients[i];
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
			const std::vector<state_gradient_pair*>& a_values,
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
			std::vector<double> l_gradients = normalized_gradients();
			for (int i = 0; i < m_values.size(); i++)
			{
				auto& l_value = m_values[i];
				auto& l_momentum = m_momenta[i];
				l_momentum = m_beta * l_momentum + m_alpha * l_gradients[i];
				l_value->m_state -= m_learn_rate * l_momentum;
			}
		}

	};

}
