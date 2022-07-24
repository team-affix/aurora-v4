#pragma once
#include "affix-base/pch.h"
#include "maths.h"

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

}
