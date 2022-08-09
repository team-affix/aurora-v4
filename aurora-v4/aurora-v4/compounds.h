#pragma once
#include "affix-base/pch.h"
#include <vector>
#include <functional>
#include "maths.h"
#include "elements.h"

namespace aurora
{
	inline std::vector<state_gradient_pair*> parameters(
		const size_t& a_count
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_count; i++)
			l_result.push_back(parameter());
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> parameters(
		const size_t& a_rows,
		const size_t& a_cols
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		for (int i = 0; i < a_rows; i++)
			l_result.push_back(parameters(a_cols));
		return l_result;
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

	inline state_gradient_pair* bias(
		state_gradient_pair* a_x
	)
	{
		return add(parameter(), a_x);
	}

	inline std::vector<state_gradient_pair*> bias(
		std::vector<state_gradient_pair*> a_x
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(bias(a_x[i]));
		return l_result;
	}

	inline std::vector<state_gradient_pair*> weight_junction(
		std::vector<state_gradient_pair*> a_x,
		const size_t& a_y_size
	)
	{
		return matrix_vector_multiply(parameters(a_y_size, a_x.size()), a_x);
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

	std::vector<std::vector<state_gradient_pair*>> lstm(
		std::vector<std::vector<state_gradient_pair*>> a_x,
		const size_t& a_y_size
	);

	inline state_gradient_pair* parameterized_interpolate(
		std::vector<state_gradient_pair*> a_x
	)
	{
		return vector_vector_multiply(normalize(parameters(a_x.size())), a_x);
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

	inline std::vector<state_gradient_pair*> leaky_relu(
		std::vector<state_gradient_pair*> a_x,
		const double& a_m
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(leaky_relu(a_x[i], a_m));
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

	inline std::vector<std::vector<state_gradient_pair*>> partition(
		std::vector<state_gradient_pair*> a_x,
		const size_t& a_bin_size
	)
	{
		assert(a_x.size() % a_bin_size == 0);

		std::vector<std::vector<state_gradient_pair*>> l_result;

		for (int i = 0; i < a_x.size(); i += a_bin_size)
		{
			std::vector<state_gradient_pair*> l_bin;

			for (int j = 0; j < a_bin_size; j++)
				l_bin.push_back(a_x[i + j]);

			l_result.push_back(l_bin);

		}

		return l_result;

	}

}
