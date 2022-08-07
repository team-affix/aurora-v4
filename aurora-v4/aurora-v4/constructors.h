#pragma once
#include "affix-base/pch.h"
#include <vector>
#include <functional>
#include "maths.h"
#include "elements.h"

namespace aurora
{
	inline state_gradient_pair* parameter(

	)
	{
		affix_base::data::ptr<element_parameter> l_element(new element_parameter());
		return &l_element->m_y;
	}

	inline state_gradient_pair* constant(
		const double& a_state
	)
	{
		affix_base::data::ptr<element_constant> l_element(new element_constant(a_state));
		return &l_element->m_y;
	}

	inline state_gradient_pair* add(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		affix_base::data::ptr<element_add> l_element(new element_add(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* subtract(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		affix_base::data::ptr<element_subtract> l_element(new element_subtract(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* multiply(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		affix_base::data::ptr<element_multiply> l_element(new element_multiply(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* divide(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		affix_base::data::ptr<element_divide> l_element(new element_divide(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* pow(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		affix_base::data::ptr<element_pow> l_element(new element_pow(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* sigmoid(
		state_gradient_pair* a_x
	)
	{
		affix_base::data::ptr<element_sigmoid> l_element(new element_sigmoid(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* tanh(
		state_gradient_pair* a_x
	)
	{
		affix_base::data::ptr<element_tanh> l_element(new element_tanh(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* leaky_relu(
		state_gradient_pair* a_x,
		const double& a_m
	)
	{
		affix_base::data::ptr<element_leaky_relu> l_element(new element_leaky_relu(a_x, a_m));
		return &l_element->m_y;
	}

	inline element_branch& branch(
		model&& a_model,
		const bool& a_enabled
	)
	{
		affix_base::data::ptr<element_branch> l_element(new element_branch(std::move(a_model), a_enabled));
		return *l_element;
	}

	inline state_gradient_pair* running_average(
		state_gradient_pair* a_x,
		double a_beta
	)
	{
		affix_base::data::ptr<element_running_average> l_element(new element_running_average(a_x, a_beta));
		return &l_element->m_y;
	}

	inline state_gradient_pair* log(
		state_gradient_pair* a_x
	)
	{
		affix_base::data::ptr<element_log> l_element(new element_log(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* additive_aggregate(
		std::vector<state_gradient_pair*> a_x
	)
	{
		assert(a_x.size() > 0);

		state_gradient_pair* l_result = a_x[0];

		for (int i = 1; i < a_x.size(); i++)
		{
			l_result = add(l_result, a_x[i]);
		}

		return l_result;

	}

	inline state_gradient_pair* vector_vector_multiply(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<state_gradient_pair*> l_multiply_ys;

		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_multiply_ys.push_back(multiply(a_x_0[i], a_x_1[i]));
		}

		return additive_aggregate(l_multiply_ys);

	}

	inline state_gradient_pair* bias(
		state_gradient_pair* a_x
	)
	{
		return add(parameter(), a_x);
	}

	inline state_gradient_pair* weight(
		state_gradient_pair* a_x
	)
	{
		return multiply(parameter(), a_x);
	}

	inline state_gradient_pair* weights(
		std::vector<state_gradient_pair*> a_x
	)
	{
		std::vector<state_gradient_pair*> l_weight_ys;

		for (int i = 0; i < a_x.size(); i++)
		{
			l_weight_ys.push_back(weight(a_x[i]));
		}

		return additive_aggregate(l_weight_ys);

	}

	inline std::vector<state_gradient_pair*> weight_junction(
		std::vector<state_gradient_pair*> a_x,
		const size_t& a_y_size
	)
	{
		std::vector<state_gradient_pair*> l_result;

		for (int i = 0; i < a_y_size; i++)
		{
			l_result.push_back(weights(a_x));
		}

		return l_result;

	}

	inline state_gradient_pair* multiplicative_aggregate(
		std::vector<state_gradient_pair*> a_x
	)
	{
		assert(a_x.size() > 0);

		state_gradient_pair* l_result = a_x[0];

		for (int i = 1; i < a_x.size(); i++)
		{
			l_result = multiply(l_result, a_x[i]);
		}

		return l_result;

	}

	inline std::vector<state_gradient_pair*> normalize(
		std::vector<state_gradient_pair*> a_x
	)
	{
		std::vector<state_gradient_pair*> l_result;

		state_gradient_pair* l_sum = additive_aggregate(a_x);

		for (int i = 0; i < a_x.size(); i++)
		{
			l_result.push_back(divide(a_x[i], l_sum));
		}

		return l_result;

	}

	inline std::vector<state_gradient_pair*> parameterized_normalize(
		const size_t& a_parameter_count
	)
	{
		std::vector<state_gradient_pair*> l_sigmoid_ys;

		for (int i = 0; i < a_parameter_count; i++)
		{
			l_sigmoid_ys.push_back(sigmoid(parameter()));
		}

		return normalize(l_sigmoid_ys);

	}

	struct tnn_layer_info
	{
		size_t m_size;
		std::function<state_gradient_pair* (state_gradient_pair*)> m_generate_neurons;

		tnn_layer_info(
			const size_t& a_size,
			const std::function<state_gradient_pair* (state_gradient_pair*)>& a_generate_neurons
		) :
			m_size(a_size),
			m_generate_neurons(a_generate_neurons)
		{

		}

	};

	inline std::vector<state_gradient_pair*> tnn(
		std::vector<state_gradient_pair*> a_x,
		std::vector<tnn_layer_info> a_layer_infos
	)
	{
		std::vector<state_gradient_pair*> l_y = a_x;

		for (int i = 0; i < a_layer_infos.size(); i++)
		{
			std::vector<state_gradient_pair*> l_w = weight_junction(l_y, a_layer_infos[i].m_size);

			l_y.resize(l_w.size());

			for (int j = 0; j < l_w.size(); j++)
				l_y[j] = a_layer_infos[i].m_generate_neurons(l_w[j]);

		}

		return l_y;

	}

	std::vector<std::vector<state_gradient_pair*>> lstm(
		std::vector<std::vector<state_gradient_pair*>> a_x,
		const size_t& a_y_size
	);

	inline state_gradient_pair* parameterized_interpolate(
		std::vector<state_gradient_pair*> a_x
	)
	{
		return vector_vector_multiply(parameterized_normalize(a_x.size()), a_x);
	}

	inline state_gradient_pair* vector_magnitude(
		std::vector<state_gradient_pair*> a_x
	)
	{
		std::vector<state_gradient_pair*> l_square_ys(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
		{
			l_square_ys[i] = pow(a_x[i], constant(2));
		}
		return pow(additive_aggregate(l_square_ys), constant(0.5));
	}

	inline state_gradient_pair* cosine_similarity(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		auto l_multiply = vector_vector_multiply(a_x_0, a_x_1);
		auto l_magnitude_0 = vector_magnitude(a_x_0);
		auto l_magnitude_1 = vector_magnitude(a_x_1);
		return divide(divide(l_multiply, l_magnitude_0), l_magnitude_1);
	}

	inline std::vector<state_gradient_pair*> vector_scalar_multiply(
		std::vector<state_gradient_pair*> a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result.push_back(multiply(a_x_0[i], a_x_1));
		}
		return l_result;
	}

	inline std::vector<state_gradient_pair*> vector_vector_add(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result.push_back(add(a_x_0[i], a_x_1[i]));
		}
		return l_result;
	}

	inline std::vector<state_gradient_pair*> vector_additive_aggregate(
		std::vector<std::vector<state_gradient_pair*>> a_x
	)
	{
		std::vector<state_gradient_pair*> l_result = a_x[0];
		for (int i = 1; i < a_x.size(); i++)
			l_result = vector_vector_add(l_result, a_x[i]);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> matrix_transpose(
		std::vector<std::vector<state_gradient_pair*>> a_x
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;

		// Resize the output matrix to have a number of rows equal to the number of 
		// columns in the input matrix
		l_result.resize(a_x[0].size());

		for (int i = 0; i < l_result.size(); i++)
		{
			// Resize each row of the output matrix to have a number of columns equal to
			// the number of rows in the input matrix
			l_result[i].resize(a_x.size());
		}

		for (int i = 0; i < a_x.size(); i++)
		{
			for (int j = 0; j < a_x[i].size(); j++)
			{
				// Link up each pointer
				l_result[j][i] = a_x[i][j];
			}
		}

		return l_result;

	}

	inline std::vector<state_gradient_pair*> matrix_vector_multiply(
		std::vector<std::vector<state_gradient_pair*>> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		assert(a_x_0[0].size() == a_x_1.size());
		auto l_transpose = matrix_transpose(a_x_0);
		std::vector<std::vector<state_gradient_pair*>> l_scaled_transpose;
		for (int i = 0; i < a_x_1.size(); i++)
		{
			l_scaled_transpose.push_back(vector_scalar_multiply(l_transpose[i], a_x_1[i]));
		}
		return vector_additive_aggregate(l_scaled_transpose);
	}

	inline std::vector<state_gradient_pair*> sigmoid(
		std::vector<state_gradient_pair*> a_x
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(sigmoid(a_x[i]));
		return l_result;
	}

	inline std::vector<state_gradient_pair*> tanh(
		std::vector<state_gradient_pair*> a_x
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(tanh(a_x[i]));
		return l_result;
	}

	inline std::vector<state_gradient_pair*> vector_vector_subtract(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result.push_back(subtract(a_x_0[i], a_x_1[i]));
		}
		return l_result;
	}

	inline state_gradient_pair* euclidian_distance(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		return vector_magnitude(vector_vector_subtract(a_x_0, a_x_1));
	}

	inline std::vector<state_gradient_pair*> similarity_interpolate(
		std::vector<state_gradient_pair*> a_query,
		std::vector<std::vector<state_gradient_pair*>> a_keys,
		std::vector<std::vector<state_gradient_pair*>> a_values
	)
	{
		std::vector<state_gradient_pair*> l_similarity_ys; // Each element between 0 and +inf

		for (int i = 0; i < a_keys.size(); i++)
		{
			auto l_distance = euclidian_distance(a_query, a_keys[i]);
			auto l_stabilized = add(l_distance, constant(0.0000001));
			l_similarity_ys.push_back(divide(constant(1), l_stabilized));
		}

		auto l_normalized = normalize(l_similarity_ys);

		auto l_transpose = matrix_transpose(a_values);

		return matrix_vector_multiply(l_transpose, l_normalized);

	}

}
