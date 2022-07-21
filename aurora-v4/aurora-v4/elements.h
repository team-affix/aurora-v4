#pragma once
#include <vector>
#include "affix-base/ptr.h"
#include "maths.h"

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
			std::vector<affix_base::data::ptr<element>>& a_elements
		)
		{
			a_elements.push_back(this);
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
	public:
		state_gradient_pair m_y;

	public:
		virtual ~parameter(

		)
		{

		}

		parameter(
			std::vector<affix_base::data::ptr<element>>& a_elements,
			std::vector<parameter*>& a_parameters
		) :
			element(a_elements)
		{
			a_parameters.push_back(this);
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			const double& a_y
		) :
			element(a_elements),
			m_y({ a_y, 0 })
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x_0,
			state_gradient_pair* a_x_1
		) :
			element(a_elements),
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x_0,
			state_gradient_pair* a_x_1
		) :
			element(a_elements),
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x_0,
			state_gradient_pair* a_x_1
		) :
			element(a_elements),
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x_0,
			state_gradient_pair* a_x_1
		) :
			element(a_elements),
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x
		) :
			element(a_elements),
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x
		) :
			element(a_elements),
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
			std::vector<affix_base::data::ptr<element>>& a_elements,
			state_gradient_pair* a_x,
			const double& a_m
		) :
			element(a_elements),
			m_x(a_x),
			m_m(a_m)
		{

		}

		virtual void fwd(

		)
		{
			if (m_x->m_state > 0)
				m_y.m_state = m_x->m_state;
			else
				m_y.m_state = m_m * m_x->m_state;
		}

		virtual void bwd(

		)
		{
			if (m_x->m_state > 0)
				m_x->m_gradient += m_y.m_gradient;
			else
				m_x->m_gradient += m_y.m_gradient * m_m;
			m_y.m_gradient = 0;
		}

	};

}


