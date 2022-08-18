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
			std::vector<state_gradient_pair*> a_values
		) :
			m_values(a_values.begin(), a_values.end())
		{

		}

		void normalize_gradients(

		)
		{
			double l_normalization_denominator = 0;

			for (auto& l_value : m_values)
				l_normalization_denominator += std::abs(l_value->m_gradient);

			for (auto& l_value : m_values)
				l_value->m_gradient /= l_normalization_denominator;

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
			m_value->m_state -= m_learn_rate * m_value->m_gradient;
			m_value->m_gradient = 0;
		}

	};

	class gradient_descent_with_momentum : public optimizer
	{
	public:
		double m_learn_rate = 0;
		double m_beta = 0;
		double m_alpha = 0;
		double m_momentum = 0;

	public:
		gradient_descent_with_momentum(
			std::vector<state_gradient_pair*> a_values,
			const double& a_learn_rate,
			const double& a_beta
		) :
			optimizer(a_values),
			m_learn_rate(a_learn_rate),
			m_beta(a_beta),
			m_alpha(1.0 - a_beta)
		{
			assert(a_beta >= 0 && a_beta <= 1);
		}

		virtual void update(

		)
		{
			m_momentum = m_beta * m_momentum + m_alpha * m_value->m_gradient;
			m_value->m_state -= m_learn_rate * m_momentum;
			m_value->m_gradient = 0;
		}

	};

}
