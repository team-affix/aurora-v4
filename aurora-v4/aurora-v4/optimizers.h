#pragma once
#include "affix-base/pch.h"

namespace aurora
{
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

	inline std::function<affix_base::data::ptr<optimizer>(affix_base::data::ptr<state_gradient_pair>)> gradient_descent(
		const double& a_learn_rate
	)
	{
		class optimizer_gradient_descent : public optimizer
		{
		public:
			double m_learn_rate = 0;

		public:
			optimizer_gradient_descent(
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
		return [a_learn_rate](
			affix_base::data::ptr<state_gradient_pair> a_parameter
		)
		{
			return new optimizer_gradient_descent(a_parameter, a_learn_rate);
		};

	}

	inline std::function<affix_base::data::ptr<optimizer>(affix_base::data::ptr<state_gradient_pair>)> gradient_descent_with_momentum(
		const double& a_learn_rate,
		const double& a_beta
	)
	{
		class optimizer_gradient_descent_with_momentum : public optimizer
		{
		public:
			double m_learn_rate = 0;
			double m_beta = 0;
			double m_momentum = 0;
			double m_alpha = 0;

		public:
			optimizer_gradient_descent_with_momentum(
				state_gradient_pair* a_value,
				const double& a_learn_rate,
				const double& a_beta
			) :
				optimizer(a_value),
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
		return [a_learn_rate, a_beta](
			affix_base::data::ptr<state_gradient_pair> a_parameter
		)
		{
			return new optimizer_gradient_descent_with_momentum(a_parameter, a_learn_rate, a_beta);
		};

	}

}
