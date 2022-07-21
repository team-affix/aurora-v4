#pragma once
#include "affix-base/pch.h"
#include <vector>

namespace aurora
{
	struct state_gradient_pair
	{
		double m_state = 0;
		double m_gradient = 0;
	};


	std::vector<state_gradient_pair*> pointers_to_each_element(
		std::vector<state_gradient_pair>& a_vec
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_vec.size(); i++)
			l_result.push_back(&a_vec[i]);
		return l_result;
	}

}
