#include "affix-base/pch.h"
#include "compounds.h"
#include "generators.h"

using namespace aurora;

struct lstm_timestep
{
public:
	std::vector<state_gradient_pair*> m_cy;
	std::vector<state_gradient_pair*> m_y;

public:
	lstm_timestep(
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

		auto l_forget_gate = tnn(l_hx_x_concat,
			{
				tnn_layer_info(a_hx.size(), neuron_sigmoid())
			});

		auto l_input_limit_gate = tnn(l_hx_x_concat,
			{
				tnn_layer_info(a_hx.size(), neuron_sigmoid())
			});

		auto l_input_gate = tnn(l_hx_x_concat,
			{
				tnn_layer_info(a_hx.size(), neuron_tanh())
			});

		auto l_output_gate = tnn(l_hx_x_concat,
			{
				tnn_layer_info(a_hx.size(), neuron_sigmoid())
			});


		std::vector<state_gradient_pair*> l_cell_state_after_forget;

		// Forget parts of the cell state
		for (int i = 0; i < l_forget_gate.size(); i++)
		{
			l_cell_state_after_forget.push_back(multiply(a_cx[i], l_forget_gate[i]));
		}

		std::vector<state_gradient_pair*> l_limited_input_ys;

		// Calculate the input to the cell state
		for (int i = 0; i < l_input_gate.size(); i++)
		{
			l_limited_input_ys.push_back(multiply(l_input_gate[i], l_input_limit_gate[i]));
		}

		std::vector<state_gradient_pair*> l_cell_state_after_input;

		// Write the input to the cell state
		for (int i = 0; i < l_limited_input_ys.size(); i++)
		{
			l_cell_state_after_input.push_back(add(l_cell_state_after_forget[i], l_limited_input_ys[i]));
		}

		// Cell state is now finalized, save it as the cell state output
		m_cy = l_cell_state_after_input;

		std::vector<state_gradient_pair*> l_cell_state_after_tanh;

		// Do a temporary step to compute tanh(cy)
		for (int i = 0; i < l_cell_state_after_input.size(); i++)
		{
			l_cell_state_after_tanh.push_back(tanh(l_cell_state_after_input[i]));
		}

		// Compute output to the timestep
		for (int i = 0; i < l_output_gate.size(); i++)
		{
			m_y.push_back(multiply(l_output_gate[i], l_cell_state_after_tanh[i]));
		}

	}

};

std::vector<std::vector<state_gradient_pair*>> aurora::lstm(
	std::vector<std::vector<state_gradient_pair*>> a_x,
	const size_t& a_y_size
)
{
	std::vector<std::vector<state_gradient_pair*>> l_result;

	std::vector<state_gradient_pair*> l_cy;
	std::vector<state_gradient_pair*> l_hy;

	// Initialize the cell state (make it learnable using parameters)
	for (int i = 0; i < a_y_size; i++)
	{
		l_cy.push_back(parameter());
	}

	// Initialize the hidden state
	for (int i = 0; i < a_y_size; i++)
	{
		l_hy.push_back(parameter());
	}


	std::vector<model> l_timestep_models;

	for (int i = 0; i < a_x.size(); i++)
	{
		model::begin();

		lstm_timestep l_timestep(a_x[i], l_cy, l_hy);
		l_cy = l_timestep.m_cy;
		l_hy = l_timestep.m_y;
		l_result.push_back(l_timestep.m_y);

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

	return l_result;

}
