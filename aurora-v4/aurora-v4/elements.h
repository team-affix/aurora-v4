#pragma once
#include "affix-base/pch.h"
#include <vector>
#include "affix-base/ptr.h"
#include "maths.h"
#include "construction.h"

namespace aurora
{
	class element
	{
	public:
		virtual ~element(

		)
		{

		}

		element(

		)
		{
			element_vector::insert(affix_base::data::ptr<element>(this));
		}

		element(
			const element&
		) = delete;

		element& operator=(
			const element&
			) = delete;

		virtual void fwd(

		)
		{

		}

		virtual void bwd(

		)
		{

		}

	};

	inline state_gradient_pair* parameter(

	)
	{
		return parameter_vector::next();
	}

	inline state_gradient_pair* constant(
		const double& a_state
	)
	{
		class element_constant : public element
		{
		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_constant(

			)
			{

			}

			element_constant(
				const double& a_y
			) :
				m_y(a_y)
			{

			}

		};

		affix_base::data::ptr<element_constant> l_element(new element_constant(a_state));
		return &l_element->m_y;
	}

	inline state_gradient_pair* add(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_add : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_add(

			)
			{

			}

			element_add(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state + m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient;
				m_x_1->m_gradient += m_y.m_gradient;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_add> l_element(new element_add(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* subtract(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_subtract : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_subtract(

			)
			{

			}

			element_subtract(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state - m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient;
				m_x_1->m_gradient -= m_y.m_gradient;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_subtract> l_element(new element_subtract(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* multiply(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_multiply : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_multiply(

			)
			{

			}

			element_multiply(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state * m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient * m_x_1->m_state;
				m_x_1->m_gradient += m_y.m_gradient * m_x_0->m_state;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_multiply> l_element(new element_multiply(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* divide(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_divide : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_divide(

			)
			{

			}

			element_divide(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state / m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient / m_x_1->m_state;
				m_x_1->m_gradient += m_y.m_gradient * (-m_x_0->m_state / std::pow(m_x_1->m_state, 2.0));
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_divide> l_element(new element_divide(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* pow(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_pow : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_pow(

			)
			{

			}

			element_pow(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::pow(m_x_0->m_state, m_x_1->m_state);
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient * m_x_1->m_state * std::pow(m_x_0->m_state, m_x_1->m_state - 1.0);
				m_x_1->m_gradient += m_y.m_gradient * std::pow(m_x_0->m_state, m_x_1->m_state) * std::log(m_x_0->m_state);
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_pow> l_element(new element_pow(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* sigmoid(
		state_gradient_pair* a_x
	)
	{
		class element_sigmoid : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			element_sigmoid(
				state_gradient_pair* a_x
			) :
				m_x(a_x)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = 1.0 / (1.0 + exp(-m_x->m_state));
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_y.m_gradient * m_y.m_state * (1.0 - m_y.m_state);
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_sigmoid> l_element(new element_sigmoid(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* tanh(
		state_gradient_pair* a_x
	)
	{
		class element_tanh : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_tanh(

			)
			{

			}

			element_tanh(
				state_gradient_pair* a_x
			) :
				m_x(a_x)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::tanh(m_x->m_state);
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_y.m_gradient / std::pow(cosh(m_x->m_state), 2.0);
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_tanh> l_element(new element_tanh(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* leaky_relu(
		state_gradient_pair* a_x,
		const double& a_m
	)
	{
		class element_leaky_relu : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;
			double m_m = 0;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_leaky_relu(

			)
			{

			}

			element_leaky_relu(
				state_gradient_pair* a_x,
				const double& a_m
			) :
				m_x(a_x),
				m_m(a_m)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state =
					(m_x->m_state > 0) * m_x->m_state +
					(m_x->m_state <= 0) * m_m * m_x->m_state;
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient +=
					(m_x->m_state > 0) * m_y.m_gradient +
					(m_x->m_state <= 0) * m_y.m_gradient * m_m;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_leaky_relu> l_element(new element_leaky_relu(a_x, a_m));
		return &l_element->m_y;
	}

	inline bool* branch(
		element_vector&& a_model,
		const bool& a_enabled
	)
	{
		class element_branch : public element
		{
		private:
			element_vector m_element_vector;

		public:
			bool m_enabled = false;

		public:
			virtual ~element_branch(

			)
			{

			}

			element_branch(
				element_vector&& a_element_vector,
				const bool& a_enabled
			) :
				m_element_vector(a_element_vector),
				m_enabled(a_enabled)
			{

			}

			virtual void fwd(

			)
			{
				if (m_enabled)
				{
					m_element_vector.fwd();
				}
			}

			virtual void bwd(

			)
			{
				if (m_enabled)
				{
					m_element_vector.bwd();
				}
			}

		};
		affix_base::data::ptr<element_branch> l_element(new element_branch(std::move(a_model), a_enabled));
		return &l_element->m_enabled;
	}

	inline state_gradient_pair* running_average(
		state_gradient_pair* a_x,
		double a_beta
	)
	{
		class element_running_average : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;
			double m_beta = 0;
			double m_alpha = 0;

		public:
			state_gradient_pair m_y = { 1 };

		public:
			virtual ~element_running_average(

			)
			{

			}

			element_running_average(
				state_gradient_pair* a_x,
				const double& a_beta
			) :
				m_x(a_x),
				m_beta(a_beta),
				m_alpha(1.0 - a_beta)
			{
				assert(a_beta >= 0 && a_beta <= 1);
			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_beta * m_y.m_state + m_alpha * m_x->m_state;
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_alpha * m_y.m_gradient;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_running_average> l_element(new element_running_average(a_x, a_beta));
		return &l_element->m_y;
	}

	inline state_gradient_pair* log(
		state_gradient_pair* a_x
	)
	{
		class element_log
		{
		private:
			state_gradient_pair* m_x = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_log(

			)
			{

			}

			element_log(
				state_gradient_pair* a_x
			) :
				m_x(a_x)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::log(m_x->m_state);
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_y.m_gradient / m_x->m_state;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_log> l_element(new element_log(a_x));
		return &l_element->m_y;
	}

}
