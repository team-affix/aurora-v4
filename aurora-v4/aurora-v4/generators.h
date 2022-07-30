#pragma once
#include "affix-base/pch.h"
#include <functional>
#include <vector>
#include "affix-base/ptr.h"
#include "maths.h"
#include "elements.h"
#include "compounds.h"

namespace aurora
{
	inline std::function<state_gradient_pair* (state_gradient_pair*)> neuron_sigmoid(

	)
	{
		return [&](state_gradient_pair* a_x)
		{
			bias l_bias(a_x);
			affix_base::data::ptr<sigmoid_activate> l_act(new sigmoid_activate(l_bias.m_y));
			return &l_act->m_y;
		};
	}

	inline std::function<state_gradient_pair* (state_gradient_pair*)> neuron_tanh(

	)
	{
		return [&](state_gradient_pair* a_x)
		{
			bias l_bias(a_x);
			affix_base::data::ptr<tanh_activate> l_act(new tanh_activate(l_bias.m_y));
			return &l_act->m_y;
		};
	}

	inline std::function<state_gradient_pair* (state_gradient_pair*)> neuron_leaky_relu(
	)
	{
		return [&](state_gradient_pair* a_x)
		{
			bias l_bias(a_x);
			affix_base::data::ptr<leaky_relu_activate> l_act(new leaky_relu_activate(l_bias.m_y, 0.3));
			return &l_act->m_y;
		};
	}

}
