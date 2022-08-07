#pragma once
#include "affix-base/pch.h"
#include <functional>
#include <vector>
#include "affix-base/ptr.h"
#include "maths.h"
#include "elements.h"
#include "constructors.h"

namespace aurora
{
	inline std::function<state_gradient_pair* (state_gradient_pair*)> neuron_sigmoid(

	)
	{
		return [&](state_gradient_pair* a_x)
		{
			return sigmoid(bias(a_x));
		};
	}

	inline std::function<state_gradient_pair* (state_gradient_pair*)> neuron_tanh(

	)
	{
		return [&](state_gradient_pair* a_x)
		{
			return tanh(bias(a_x));
		};
	}

	inline std::function<state_gradient_pair* (state_gradient_pair*)> neuron_leaky_relu(
	)
	{
		return [&](state_gradient_pair* a_x)
		{
			return leaky_relu(bias(a_x), 0.3);
		};
	}

}
