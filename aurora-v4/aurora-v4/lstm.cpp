#include "affix-base/pch.h"
#include "compounds.h"
#include "generators.h"

using namespace aurora;

lstm::timestep::timestep(
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

	tnn l_forget_gate(l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_sigmoid())
		});

	tnn l_input_limit_gate(l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_sigmoid())
		});

	tnn l_input_gate(l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_tanh())
		});

	tnn l_output_gate(l_hx_x_concat,
		{
			tnn::layer_info(a_hx.size(), neuron_sigmoid())
		});


	std::vector<state_gradient_pair*> l_cell_state_after_forget;

	// Forget parts of the cell state
	for (int i = 0; i < l_forget_gate.m_y.size(); i++)
	{
		affix_base::data::ptr<multiply> l_multiply(new multiply(a_cx[i], l_forget_gate.m_y[i]));
		l_cell_state_after_forget.push_back(&l_multiply->m_y);
	}

	std::vector<state_gradient_pair*> l_limited_input_ys;

	// Calculate the input to the cell state
	for (int i = 0; i < l_input_gate.m_y.size(); i++)
	{
		affix_base::data::ptr<multiply> l_multiply(new multiply(l_input_gate.m_y[i], l_input_limit_gate.m_y[i]));
		l_limited_input_ys.push_back(&l_multiply->m_y);
	}

	std::vector<state_gradient_pair*> l_cell_state_after_input;

	// Write the input to the cell state
	for (int i = 0; i < l_limited_input_ys.size(); i++)
	{
		affix_base::data::ptr<add> l_add(new add(l_cell_state_after_forget[i], l_limited_input_ys[i]));
		l_cell_state_after_input.push_back(&l_add->m_y);
	}

	// Cell state is now finalized, save it as the cell state output
	m_cy = l_cell_state_after_input;
	
	std::vector<state_gradient_pair*> l_cell_state_after_tanh;

	// Do a temporary step to compute tanh(cy)
	for (int i = 0; i < l_cell_state_after_input.size(); i++)
	{
		affix_base::data::ptr<tanh_activate> l_tanh(new tanh_activate(l_cell_state_after_input[i]));
		l_cell_state_after_tanh.push_back(&l_tanh->m_y);
	}

	// Compute output to the timestep
	for (int i = 0; i < l_output_gate.m_y.size(); i++)
	{
		affix_base::data::ptr<multiply> l_multiply(new multiply(l_output_gate.m_y[i], l_cell_state_after_tanh[i]));
		m_y.push_back(&l_multiply->m_y);
	}

}

lstm::lstm(
	std::vector<std::vector<state_gradient_pair*>> a_x,
	const size_t& a_y_size
)
{
	std::vector<state_gradient_pair*> l_cy;
	std::vector<state_gradient_pair*> l_hy;

	// Initialize the cell state (make it learnable using parameters)
	for (int i = 0; i < a_y_size; i++)
	{
		affix_base::data::ptr<parameter> l_parameter(new parameter());
		l_cy.push_back(&l_parameter->m_y);
	}

	// Initialize the hidden state
	for (int i = 0; i < a_y_size; i++)
	{
		affix_base::data::ptr<parameter> l_parameter(new parameter());
		l_hy.push_back(&l_parameter->m_y);
	}


	std::vector<model> l_timestep_models;

	for (int i = 0; i < a_x.size(); i++)
	{
		model::begin();

		lstm::timestep l_timestep(a_x[i], l_cy, l_hy);
		l_cy = l_timestep.m_cy;
		l_hy = l_timestep.m_y;
		m_y.push_back(l_timestep.m_y);
		
		// Push the LSTM Timestep model into the list
		l_timestep_models.push_back(model::end());

	}

	model& l_initial_timestep_model = l_timestep_models[0];

	for (int i = 1; i < l_timestep_models.size(); i++)
	{
		for (int j = 0; j < l_timestep_models[i].parameters().size(); j++)
		{
			// Link all the parameters from each timestep together
			l_timestep_models[i].parameters()[j].group_link(l_initial_timestep_model.parameters()[j]);
		}
	}

	for (int i = 0; i < l_timestep_models.size(); i++)
	{
		// Insert every element from every timestep into the main model
		model::insert(l_timestep_models[i].elements());
	}

	// Add the parameters which each timestep now shares to the list of all parameters
	model::insert(l_initial_timestep_model.parameters());

}
