#pragma once
#include "affix-base/pch.h"
#include <vector>
#include <functional>
#include "maths.h"
#include "elements.h"

namespace aurora
{
	struct additive_aggregate
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		additive_aggregate(
			std::vector<state_gradient_pair*> a_x
		)
		{
			assert(a_x.size() > 0);

			m_y = a_x[0];

			for (int i = 1; i < a_x.size(); i++)
			{
				affix_base::data::ptr<add> l_add(new add(m_y, a_x[i]));
				m_y = &l_add->m_y;
			}

		}

	};

	struct vector_vector_multiply
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		vector_vector_multiply(
			std::vector<state_gradient_pair*> a_x_0,
			std::vector<state_gradient_pair*> a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<state_gradient_pair*> l_multiply_ys;

			for (int i = 0; i < a_x_0.size(); i++)
			{
				affix_base::data::ptr<multiply> l_multiply(new multiply(a_x_0[i], a_x_1[i]));
				l_multiply_ys.push_back(&l_multiply->m_y);
			}

			additive_aggregate l_additive_aggregate(l_multiply_ys);

			m_y = l_additive_aggregate.m_y;

		}

	};

	struct bias
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		virtual ~bias(

		)
		{

		}

		bias(
			state_gradient_pair* a_x
		)
		{
			affix_base::data::ptr<parameter> l_parameter(new parameter());
			affix_base::data::ptr<add> l_add(new add(a_x, &l_parameter->m_y));
			m_y = &l_add->m_y;
		}

	};

	struct weight
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		virtual ~weight(

		)
		{

		}

		weight(
			state_gradient_pair* a_x
		)
		{
			affix_base::data::ptr<parameter> l_parameter(new parameter());
			affix_base::data::ptr<multiply> l_multiply(new multiply(a_x, &l_parameter->m_y));
			m_y = &l_multiply->m_y;
		}

	};

	struct weights
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		virtual ~weights(

		)
		{

		}

		weights(
			std::vector<state_gradient_pair*> a_x
		)
		{
			std::vector<state_gradient_pair*> l_weight_ys;

			for (int i = 0; i < a_x.size(); i++)
			{
				affix_base::data::ptr<weight> l_weight(new weight(a_x[i]));
				l_weight_ys.push_back(l_weight->m_y);
			}

			additive_aggregate l_agg(l_weight_ys);

			m_y = l_agg.m_y;

		}

	};

	struct weight_junction
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		weight_junction(
			std::vector<state_gradient_pair*> a_x,
			const size_t& a_y_size
		)
		{
			for (int i = 0; i < a_y_size; i++)
			{
				weights l_weights(a_x);
				m_y.push_back(l_weights.m_y);
			}
		}

	};

	struct multiplicative_aggregate
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		multiplicative_aggregate(
			std::vector<state_gradient_pair*> a_x
		)
		{
			assert(a_x.size() > 0);

			m_y = a_x[0];

			for (int i = 1; i < a_x.size(); i++)
			{
				affix_base::data::ptr<multiply> l_multiply(new multiply(m_y, a_x[i]));
				m_y = &l_multiply->m_y;
			}
		}

	};

	struct normalize
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		normalize(
			std::vector<state_gradient_pair*> a_x
		)
		{
			additive_aggregate l_additive_aggregate(a_x);
			for (int i = 0; i < a_x.size(); i++)
			{
				affix_base::data::ptr<divide> l_divide(new divide(a_x[i], l_additive_aggregate.m_y));
				m_y.push_back(&l_divide->m_y);
			}
		}

	};

	struct parameterized_normalize
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		parameterized_normalize(
			const size_t& a_parameter_count
		)
		{
			std::vector<state_gradient_pair*> l_parameter_ys;

			for (int i = 0; i < a_parameter_count; i++)
			{
				affix_base::data::ptr<parameter> l_parameter(new parameter());
				l_parameter_ys.push_back(&l_parameter->m_y);
			}

			std::vector<state_gradient_pair*> l_sigmoid_ys;

			for (int i = 0; i < a_parameter_count; i++)
			{
				affix_base::data::ptr<sigmoid_activate> l_sigmoid_activate(new sigmoid_activate(l_parameter_ys[i]));
				l_sigmoid_ys.push_back(&l_sigmoid_activate->m_y);
			}

			normalize l_normalize(l_sigmoid_ys);

			m_y = l_normalize.m_y;

		}

	};

	struct tnn
	{
	public:
		struct layer_info
		{
			size_t m_size;
			std::function<state_gradient_pair* (state_gradient_pair*)> m_generate_neurons;

			layer_info(
				const size_t& a_size,
				const std::function<state_gradient_pair* (state_gradient_pair*)>& a_generate_neurons
			) :
				m_size(a_size),
				m_generate_neurons(a_generate_neurons)
			{

			}

		};

	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		tnn(
			std::vector<state_gradient_pair*> a_x,
			std::vector<tnn::layer_info> a_layer_infos
		)
		{
			std::vector<state_gradient_pair*> l_y = a_x;

			for (int i = 0; i < a_layer_infos.size(); i++)
			{
				weight_junction l_w(l_y, a_layer_infos[i].m_size);
				
				l_y.resize(l_w.m_y.size());

				for (int j = 0; j < l_w.m_y.size(); j++)
					l_y[j] = a_layer_infos[i].m_generate_neurons(l_w.m_y[j]);

			}

			m_y = l_y;
		}

	};

	struct lstm
	{
	private:
		struct timestep
		{
		public:
			std::vector<state_gradient_pair*> m_cy;
			std::vector<state_gradient_pair*> m_y;

		public:
			timestep(
				std::vector<state_gradient_pair*> a_x,
				std::vector<state_gradient_pair*> a_cx,
				std::vector<state_gradient_pair*> a_hx
			);

		};

	public:
		std::vector<std::vector<state_gradient_pair*>> m_y;

	public:
		lstm(
			std::vector<std::vector<state_gradient_pair*>> a_x,
			const size_t& a_y_size
		);

	};

	struct parameterized_interpolate
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		parameterized_interpolate(
			std::vector<state_gradient_pair*> a_x
		)
		{
			parameterized_normalize l_normalize(a_x.size());

			vector_vector_multiply l_dot(a_x, l_normalize.m_y);

			m_y = l_dot.m_y;

		}

	};

	struct vector_magnitude
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		vector_magnitude(
			std::vector<state_gradient_pair*> a_x
		)
		{
			std::vector<state_gradient_pair*> l_square_ys(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
			{
				affix_base::data::ptr<constant> l_constant(new constant(2.0));
				affix_base::data::ptr<power> l_square(new power(a_x[i], &l_constant->m_y));
				l_square_ys[i] = &l_square->m_y;
			}
			additive_aggregate l_agg(l_square_ys);
			affix_base::data::ptr<constant> l_half(new constant(0.5));
			affix_base::data::ptr<power> l_sqrt(new power(l_agg.m_y, &l_half->m_y));
			m_y = &l_sqrt->m_y;
		}

	};

	struct cosine_similarity
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		cosine_similarity(
			std::vector<state_gradient_pair*> a_x_0,
			std::vector<state_gradient_pair*> a_x_1
		)
		{
			vector_vector_multiply l_multiply(a_x_0, a_x_1);
			vector_magnitude l_magnitude_0(a_x_0);
			vector_magnitude l_magnitude_1(a_x_1);
			affix_base::data::ptr<divide> l_divide_0(new divide(l_multiply.m_y, l_magnitude_0.m_y));
			affix_base::data::ptr<divide> l_divide_1(new divide(&l_divide_0->m_y, l_magnitude_1.m_y));
			m_y = &l_divide_1->m_y;
		}

	};

	struct vector_scalar_multiply
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		vector_scalar_multiply(
			std::vector<state_gradient_pair*> a_x_0,
			state_gradient_pair* a_x_1
		)
		{
			for (int i = 0; i < a_x_0.size(); i++)
			{
				affix_base::data::ptr<multiply> l_multiply(new multiply(a_x_0[i], a_x_1));
				m_y.push_back(&l_multiply->m_y);
			}
		}

	};

	struct vector_vector_add
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		vector_vector_add(
			std::vector<state_gradient_pair*> a_x_0,
			std::vector<state_gradient_pair*> a_x_1
		)
		{
			for (int i = 0; i < a_x_0.size(); i++)
			{
				affix_base::data::ptr<add> l_add(new add(a_x_0[i], a_x_1[i]));
				m_y.push_back(&l_add->m_y);
			}
		}

	};

	struct vector_additive_aggregate
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		vector_additive_aggregate(
			std::vector<std::vector<state_gradient_pair*>> a_x
		)
		{
			std::vector<state_gradient_pair*> l_y = a_x[0];
			for (int i = 1; i < a_x.size(); i++)
			{
				vector_vector_add l_add(l_y, a_x[i]);
				l_y = l_add.m_y;
			}
			m_y = l_y;
		}

	};

	struct matrix_transpose
	{
	public:
		std::vector<std::vector<state_gradient_pair*>> m_y;

	public:
		matrix_transpose(
			std::vector<std::vector<state_gradient_pair*>> a_x
		)
		{
			// Resize the output matrix to have a number of rows equal to the number of 
			// columns in the input matrix
			m_y.resize(a_x[0].size());

			for (int i = 0; i < m_y.size(); i++)
			{
				// Resize each row of the output matrix to have a number of columns equal to
				// the number of rows in the input matrix
				m_y[i].resize(a_x.size());
			}

			for (int i = 0; i < a_x.size(); i++)
			{
				for (int j = 0; j < a_x[i].size(); j++)
				{
					// Link up each pointer
					m_y[j][i] = a_x[i][j];
				}
			}
		}

	};

	struct matrix_vector_multiply
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		matrix_vector_multiply(
			std::vector<std::vector<state_gradient_pair*>> a_x_0,
			std::vector<state_gradient_pair*> a_x_1
		)
		{
			matrix_transpose l_transpose(a_x_0);
			std::vector<std::vector<state_gradient_pair*>> l_scaled_transpose;
			for (int i = 0; i < a_x_1.size(); i++)
			{
				vector_scalar_multiply l_multiply(l_transpose.m_y[i], a_x_1[i]);
				l_scaled_transpose.push_back(l_multiply.m_y);
			}
			vector_additive_aggregate l_aggregate(l_scaled_transpose);
			m_y = l_aggregate.m_y;
		}

	};

	struct vector_sigmoid
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		vector_sigmoid(
			std::vector<state_gradient_pair*> a_x
		)
		{
			for (int i = 0; i < a_x.size(); i++)
			{
				affix_base::data::ptr<sigmoid_activate> l_sigmoid(a_x[i]);
				m_y.push_back(&l_sigmoid->m_y);
			}
		}

	};

	struct vector_vector_subtract
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		vector_vector_subtract(
			std::vector<state_gradient_pair*> a_x_0,
			std::vector<state_gradient_pair*> a_x_1
		)
		{
			for (int i = 0; i < a_x_0.size(); i++)
			{
				affix_base::data::ptr<subtract> l_subtract(new subtract(a_x_0[i], a_x_1[i]));
				m_y.push_back(&l_subtract->m_y);
			}
		}

	};

	struct euclidian_distance
	{
	public:
		state_gradient_pair* m_y = nullptr;

	public:
		euclidian_distance(
			std::vector<state_gradient_pair*> a_x_0,
			std::vector<state_gradient_pair*> a_x_1
		)
		{
			vector_vector_subtract l_subtract(a_x_0, a_x_1);
			vector_magnitude l_magnitude(l_subtract.m_y);
			m_y = l_magnitude.m_y;
		}

	};

	struct similarity_interpolate
	{
	public:
		std::vector<state_gradient_pair*> m_y;

	public:
		similarity_interpolate(
			std::vector<state_gradient_pair*> a_query,
			std::vector<std::vector<state_gradient_pair*>> a_keys,
			std::vector<std::vector<state_gradient_pair*>> a_values
		)
		{
			std::vector<state_gradient_pair*> l_similarity_ys; // Each element between 0 and +inf

			for (int i = 0; i < a_keys.size(); i++)
			{
				euclidian_distance l_distance(a_query, a_keys[i]);
				affix_base::data::ptr<constant> l_numerical_stabilizer(new constant(0.0000001));
				affix_base::data::ptr<add> l_stabilized_distance(new add(l_distance.m_y, &l_numerical_stabilizer->m_y));
				affix_base::data::ptr<constant> l_one(new constant(1.0));
				affix_base::data::ptr<divide> l_reciprocal_distance(new divide(&l_one->m_y, &l_stabilized_distance->m_y));
				l_similarity_ys.push_back(&l_reciprocal_distance->m_y);
			}
			
			normalize l_normalize(l_similarity_ys);

			matrix_transpose l_transpose(a_values);

			matrix_vector_multiply l_linear_combination(l_transpose.m_y, l_normalize.m_y);

			m_y = l_linear_combination.m_y;

		}

	};

}
