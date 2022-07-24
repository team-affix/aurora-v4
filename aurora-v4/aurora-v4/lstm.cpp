#include "affix-base/pch.h"
#include "compounds.h"
#include "generators.h"

using namespace aurora;

lstm::timestep::timestep(
	std::vector<affix_base::data::ptr<element>>& a_elements,
	std::vector<affix_base::data::ptr<state_gradient_pair>>& a_parameters,
	std::vector<state_gradient_pair*> a_x,
	std::vector<state_gradient_pair*> a_cx,
	std::vector<state_gradient_pair*> a_hx
)
{
	std::vector<state_gradient_pair*> l_hx_x_concat;

	// Concatenate hx and x into one vector
	for (int i = 0; i < a_hx.size(); i++)
		l_hx_x_concat.push_back(a_hx[i]);
	for (int i = 0; i < a_x.size(); i++)
		l_hx_x_concat.push_back(a_x[i]);


	// Construct gates

	tnn l_forget_gate(a_elements, a_parameters, l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_sigmoid(a_elements, a_parameters))
		});

	tnn l_input_limit_gate(a_elements, a_parameters, l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_sigmoid(a_elements, a_parameters))
		});

	tnn l_input_gate(a_elements, a_parameters, l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_tanh(a_elements, a_parameters))
		});

	tnn l_output_gate(a_elements, a_parameters, l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_sigmoid(a_elements, a_parameters))
		});


	std::vector<state_gradient_pair*> l_cell_state_after_forget;

	// Forget parts of the cell state
	for (int i = 0; i < l_forget_gate.m_y.size(); i++)
	{
		affix_base::data::ptr<multiply> l_multiply(new multiply(a_elements, a_cx[i], l_forget_gate.m_y[i]));
		l_cell_state_after_forget.push_back(&l_multiply->m_y);
	}

	std::vector<state_gradient_pair*> l_limited_input_ys;

	// Calculate the input to the cell state
	for (int i = 0; i < l_input_gate.m_y.size(); i++)
	{
		affix_base::data::ptr<multiply> l_multiply(new multiply(a_elements, l_input_gate.m_y[i], l_input_limit_gate.m_y[i]));
		l_limited_input_ys.push_back(&l_multiply->m_y);
	}

	std::vector<state_gradient_pair*> l_cell_state_after_input;

	// Write the input to the cell state
	for (int i = 0; i < l_limited_input_ys.size(); i++)
	{
		affix_base::data::ptr<add> l_add(new add(a_elements, l_cell_state_after_forget[i], l_limited_input_ys[i]));
		l_cell_state_after_input.push_back(&l_add->m_y);
	}

	// Cell state is now finalized, save it as the cell state output
	m_cy = l_cell_state_after_input;

	
	std::vector<state_gradient_pair*> l_cell_state_after_tanh;

	// Do a temporary step to compute tanh(cy)
	for (int i = 0; i < l_cell_state_after_input.size(); i++)
	{
		affix_base::data::ptr<tanh_activate> l_tanh(new tanh_activate(a_elements, l_cell_state_after_input[i]));
		l_cell_state_after_tanh.push_back(&l_tanh->m_y);
	}

	// Compute output to the timestep
	for (int i = 0; i < l_output_gate.m_y.size(); i++)
	{
		affix_base::data::ptr<multiply> l_multiply(new multiply(a_elements, l_output_gate.m_y[i], l_cell_state_after_tanh[i]));
		m_y.push_back(&l_multiply->m_y);
	}

}
