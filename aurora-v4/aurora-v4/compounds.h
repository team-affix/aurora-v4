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

}
