#pragma once
#include <functional>
#include <vector>
#include "affix-base/ptr.h"
#include "maths.h"
#include "elements.h"
#include "compounds.h"

namespace aurora
{
	std::function<state_gradient_pair* (state_gradient_pair*)> neuron_sigmoid(
		std::vector<affix_base::data::ptr<element>>& a_elements,
		std::vector<parameter*>& a_parameters
	)
	{
		return [&](state_gradient_pair* a_x)
		{
			bias l_bias(a_elements, a_parameters, a_x);
			affix_base::data::ptr<sigmoid_activate> l_act(new sigmoid_activate(a_elements, l_bias.m_y));
			return &l_act->m_y;
		};
	}

	std::function<state_gradient_pair* (state_gradient_pair*)> neuron_tanh(
		std::vector<affix_base::data::ptr<element>>& a_elements,
		std::vector<parameter*>& a_parameters
	)
	{
		return [&](state_gradient_pair* a_x)
		{
			bias l_bias(a_elements, a_parameters, a_x);
			affix_base::data::ptr<tanh_activate> l_act(new tanh_activate(a_elements, l_bias.m_y));
			return &l_act->m_y;
		};
	}

	std::function<state_gradient_pair* (state_gradient_pair*)> neuron_leaky_relu(
		std::vector<affix_base::data::ptr<element>>& a_elements,
		std::vector<parameter*>& a_parameters
	)
	{
		return [&](state_gradient_pair* a_x)
		{
			bias l_bias(a_elements, a_parameters, a_x);
			affix_base::data::ptr<leaky_relu_activate> l_act(new leaky_relu_activate(a_elements, l_bias.m_y, 0.3));
			return &l_act->m_y;
		};
	}

}
