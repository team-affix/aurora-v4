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
	sgp_ptr_vector m_cy;
	sgp_ptr_vector m_y;

public:
	lstm_timestep(
		sgp_ptr_vector a_x,
		sgp_ptr_vector a_cx,
		sgp_ptr_vector a_hx
	)
	{
		sgp_ptr_vector l_hx_x_concat = concat(a_hx, a_x);

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
		sgp_ptr_vector l_cell_state_after_forget = hadamard(a_cx, l_forget_gate);

		// Calculate the input to the cell state
		sgp_ptr_vector l_limited_input = hadamard(l_input_gate, l_input_limit_gate);

		// Write the input to the cell state
		sgp_ptr_vector l_cell_state_after_input = add(l_cell_state_after_forget, l_limited_input);

		// Cell state is now finalized, save it as the cell state output
		m_cy = l_cell_state_after_input;

		// Do a temporary step to compute tanh(cy)
		sgp_ptr_vector l_cell_state_after_tanh = tanh(l_cell_state_after_input);

		// Compute output to the timestep
		m_y = hadamard(l_output_gate, l_cell_state_after_tanh);

	}

};

sgp_ptr_matrix aurora::latent::lstm(
	const sgp_ptr_matrix& a_x,
	const size_t& a_y_size
)
{
	sgp_ptr_matrix l_result(a_x.size());

	sgp_ptr_vector l_cy = parameters(a_y_size);
	sgp_ptr_vector l_hy = parameters(a_y_size);

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
