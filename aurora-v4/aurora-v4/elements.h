#pragma once
#include "affix-base/pch.h"
#include <vector>
#include "affix-base/ptr.h"
#include "maths.h"
#include "model.h"

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
			model::insert(affix_base::data::ptr<element>(this));
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

	class parameter : public element
	{
	private:
		affix_base::data::ptr<state_gradient_pair> m_linkable_value = new state_gradient_pair();

	public:
		state_gradient_pair m_y;

	public:
		virtual ~parameter(

		)
		{

		}

		parameter(

		) :
			m_linkable_value(new state_gradient_pair())
		{
			model::insert(m_linkable_value);
		}

		virtual void fwd(

		)
		{
			m_y.m_state = m_linkable_value->m_state;
		}

		virtual void bwd(

		)
		{
			m_linkable_value->m_gradient += m_y.m_gradient;
			m_y.m_gradient = 0;
		}

	};

	class constant : public element
	{
	public:
		state_gradient_pair m_y;

	public:
		virtual ~constant(

		)
		{

		}

		constant(
			const double& a_y
		) :
			m_y(a_y)
		{

		}

	};

	class add : public element
	{
	private:
		state_gradient_pair* m_x_0 = nullptr;
		state_gradient_pair* m_x_1 = nullptr;

	public:
		state_gradient_pair m_y;

	public:
		virtual ~add(

		)
		{

		}

		add(
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

	class multiply : public element
	{
	private:
		state_gradient_pair* m_x_0 = nullptr;
		state_gradient_pair* m_x_1 = nullptr;

	public:
		state_gradient_pair m_y;

	public:
		virtual ~multiply(

		)
		{

		}

		multiply(
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

	class divide : public element
	{
	private:
		state_gradient_pair* m_x_0 = nullptr;
		state_gradient_pair* m_x_1 = nullptr;

	public:
		state_gradient_pair m_y;

	public:
		virtual ~divide(

		)
		{

		}

		divide(
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
			m_x_1->m_gradient += m_y.m_gradient * (-m_x_0->m_state / pow(m_x_1->m_state, 2.0));
			m_y.m_gradient = 0;
		}

	};

	class power : public element
	{
	private:
		state_gradient_pair* m_x_0 = nullptr;
		state_gradient_pair* m_x_1 = nullptr;

	public:
		state_gradient_pair m_y;

	public:
		virtual ~power(

		)
		{

		}

		power(
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
			m_y.m_state = pow(m_x_0->m_state, m_x_1->m_state);
		}

		virtual void bwd(

		)
		{
			m_x_0->m_gradient += m_y.m_gradient * m_x_1->m_state * pow(m_x_0->m_state, m_x_1->m_state - 1.0);
			m_x_1->m_gradient += m_y.m_gradient * pow(m_x_0->m_state, m_x_1->m_state) * log(m_x_0->m_state);
			m_y.m_gradient = 0;
		}

	};

	class sigmoid_activate : public element
	{
	private:
		state_gradient_pair* m_x = nullptr;

	public:
		state_gradient_pair m_y;

	public:
		sigmoid_activate(
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

	class tanh_activate : public element
	{
	private:
		state_gradient_pair* m_x = nullptr;

	public:
		state_gradient_pair m_y;

	public:
		virtual ~tanh_activate(

		)
		{
			
		}

		tanh_activate(
			state_gradient_pair* a_x
		) :
			m_x(a_x)
		{

		}

		virtual void fwd(

		)
		{
			m_y.m_state = tanh(m_x->m_state);
		}

		virtual void bwd(

		)
		{
			m_x->m_gradient += m_y.m_gradient / pow(cosh(m_x->m_state), 2.0);
			m_y.m_gradient = 0;
		}

	};

	class leaky_relu_activate : public element
	{
	private:
		state_gradient_pair* m_x = nullptr;
		double m_m = 0;

	public:
		state_gradient_pair m_y;

	public:
		virtual ~leaky_relu_activate(

		)
		{

		}

		leaky_relu_activate(
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

	class branch : public element
	{
	private:
		model m_model;
		bool m_enabled = false;

	public:
		branch(
			model&& a_model,
			const bool& a_enabled
		) :
			m_model(a_model),
			m_enabled(a_enabled)
		{
			// Make sure to insert the parameters of this branch into the most external model.
			model::insert(a_model.parameters());
		}

		virtual void fwd(

		)
		{
			if (m_enabled)
			{
				m_model.fwd();
			}
		}

		virtual void bwd(

		)
		{
			if (m_enabled)
			{
				m_model.bwd();
			}
		}

		virtual void enable(

		)
		{
			m_enabled = true;
		}

		virtual void disable(

		)
		{
			m_enabled = false;
		}

		model& internal_model(

		)
		{
			return m_model;
		}

	};

}
