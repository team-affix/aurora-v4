#include "latent.h"

using namespace aurora::latent;

std::vector<element_vector> element_vector::s_element_vectors;

std::vector<parameter_vector> parameter_vector::s_parameter_vectors;

void element_vector::start(

)
{
	s_element_vectors.push_back(element_vector());
}

element_vector element_vector::stop(

)
{
	element_vector l_result = current_element_vector();
	s_element_vectors.pop_back();
	return l_result;
}

void element_vector::fwd(

)
{
	for (int i = 0; i < size(); i++)
		at(i)->fwd();
}

void element_vector::bwd(

)
{
	for (int i = size() - 1; i >= 0; i--)
		at(i)->bwd();
}

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
		std::vector<state_gradient_pair*> l_hx_x_concat = concat(a_hx, a_x);

		// Construct gates

		auto l_forget_gate = weight_junction(l_hx_x_concat, a_hx.size());
		l_forget_gate = bias(l_forget_gate);
		l_forget_gate = sigmoid(l_forget_gate);

		auto l_input_limit_gate = weight_junction(l_hx_x_concat, a_hx.size());
		l_input_limit_gate = bias(l_input_limit_gate);
		l_input_limit_gate = sigmoid(l_input_limit_gate);

		auto l_input_gate = weight_junction(l_hx_x_concat, a_hx.size());
		l_input_gate = bias(l_input_gate);
		l_input_gate = tanh(l_input_gate);

		auto l_output_gate = weight_junction(l_hx_x_concat, a_hx.size());
		l_output_gate = bias(l_output_gate);
		l_output_gate = sigmoid(l_output_gate);


		// Forget parts of the cell state
		std::vector<state_gradient_pair*> l_cell_state_after_forget = hadamard(a_cx, l_forget_gate);

		// Calculate the input to the cell state
		std::vector<state_gradient_pair*> l_limited_input = hadamard(l_input_gate, l_input_limit_gate);

		// Write the input to the cell state
		std::vector<state_gradient_pair*> l_cell_state_after_input = add(l_cell_state_after_forget, l_limited_input);

		// Cell state is now finalized, save it as the cell state output
		m_cy = l_cell_state_after_input;

		// Do a temporary step to compute tanh(cy)
		std::vector<state_gradient_pair*> l_cell_state_after_tanh = tanh(l_cell_state_after_input);

		// Compute output to the timestep
		m_y = hadamard(l_output_gate, l_cell_state_after_tanh);

	}

};

std::vector<std::vector<state_gradient_pair*>> aurora::latent::lstm(
	const std::vector<std::vector<state_gradient_pair*>>& a_x,
	const size_t& a_y_size
)
{
	std::vector<std::vector<state_gradient_pair*>> l_result(a_x.size());

	std::vector<state_gradient_pair*> l_cy = parameters(a_y_size);
	std::vector<state_gradient_pair*> l_hy = parameters(a_y_size);

	size_t l_timestep_parameters_start_index = parameter_vector::next_index();

	for (int i = 0; i < a_x.size(); i++)
	{
		parameter_vector::next_index(l_timestep_parameters_start_index);
		lstm_timestep l_timestep(a_x[i], l_cy, l_hy);
		l_cy = l_timestep.m_cy;
		l_hy = l_timestep.m_y;
		l_result[i] = l_timestep.m_y;
	}

	return l_result;

}
