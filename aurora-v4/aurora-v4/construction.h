#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"
#include "maths.h"
#include "randomization.h"

namespace aurora
{
	class element;

	struct element_vector : public std::vector<affix_base::data::ptr<element>>
	{
	private:
		static std::vector<element_vector> s_element_vectors;

	private:
		element_vector(

		)
		{

		}

		static element_vector& current_element_vector(

		)
		{
			if (s_element_vectors.size() == 0)
				throw std::exception("Error: there are currently no element vectors being constructed.");
			return s_element_vectors.back();
		}

	public:
		static void start(

		);

		static element_vector stop(

		);

		static void insert(
			const affix_base::data::ptr<element>& a_element
		)
		{
			current_element_vector().push_back(a_element);
		}

	public:
		void fwd(

		);

		void bwd(

		);

	};

	struct parameter_vector : public std::vector<affix_base::data::ptr<state_gradient_pair>>
	{
	private:
		static std::vector<parameter_vector> s_parameter_vectors;

	private:
		size_t m_next_index = 0;

	private:
		parameter_vector(

		)
		{

		}

		static parameter_vector& current_parameter_vector(

		)
		{
			if (s_parameter_vectors.size() == 0)
				throw std::exception("Error: there are currently no parameter vectors being constructed.");
			return s_parameter_vectors.back();
		}

	public:
		static void start(

		)
		{
			s_parameter_vectors.push_back({});
		}

		static parameter_vector stop(

		)
		{
			parameter_vector l_result = s_parameter_vectors.back();
			s_parameter_vectors.pop_back();
			return l_result;
		}

		static parameter_vector stop(
			const double& a_parameter_minimum_value,
			const double& a_parameter_maximum_value
		)
		{
			parameter_vector l_result = stop();

			std::uniform_real_distribution<double> l_urd(a_parameter_minimum_value, a_parameter_maximum_value);
			for (auto& l_parameter : l_result)
				l_parameter->m_state = l_urd(i_default_random_engine);

			return l_result;

		}

		static void next_index(
			const size_t& a_next_index
		)
		{
			parameter_vector& l_current_parameter_vector = current_parameter_vector();
			if (a_next_index > l_current_parameter_vector.size())
				throw std::exception("Error: a_parameter_index was out of legal bounds given s_parameter_vector's size.");
			l_current_parameter_vector.m_next_index = a_next_index;
		}

		static size_t next_index(

		)
		{
			return current_parameter_vector().m_next_index;
		}

		static state_gradient_pair* next(

		)
		{
			parameter_vector& l_current_parameter_vector = current_parameter_vector();
			if (l_current_parameter_vector.m_next_index == l_current_parameter_vector.size())
			{
				l_current_parameter_vector.push_back(new state_gradient_pair());
				l_current_parameter_vector.m_next_index++;
				return l_current_parameter_vector.back();
			}
			else
			{
				affix_base::data::ptr<state_gradient_pair> l_result = l_current_parameter_vector.at(l_current_parameter_vector.m_next_index);
				l_current_parameter_vector.m_next_index++;
				return l_result;
			}
		}

		operator std::vector<state_gradient_pair*>(

		)
		{
			std::vector<state_gradient_pair*> l_result(size());

			for (int i = 0; i < size(); i++)
				l_result[i] = at(i);

			return l_result;

		}

	};

	class element
	{
	public:
		virtual ~element(

		)
		{

		}

		element(

		)
		{
			element_vector::insert(affix_base::data::ptr<element>(this));
		}

		element(
			const element&
		) = delete;

		element& operator=(
			const element&
			) = delete;

		virtual void fwd(

		)
		{

		}

		virtual void bwd(

		)
		{

		}

	};

	inline state_gradient_pair* constant(
		const double& a_state
	)
	{
		class element_constant : public element
		{
		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_constant(

			)
			{

			}

			element_constant(
				const double& a_y
			) :
				m_y(a_y)
			{

			}

		};

		affix_base::data::ptr<element_constant> l_element(new element_constant(a_state));
		return &l_element->m_y;
	}

	inline state_gradient_pair* add(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_add : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_add(

			)
			{

			}

			element_add(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state + m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient;
				m_x_1->m_gradient += m_y.m_gradient;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_add> l_element(new element_add(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* subtract(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_subtract : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_subtract(

			)
			{

			}

			element_subtract(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state - m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient;
				m_x_1->m_gradient -= m_y.m_gradient;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_subtract> l_element(new element_subtract(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* multiply(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_multiply : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_multiply(

			)
			{

			}

			element_multiply(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state * m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient * m_x_1->m_state;
				m_x_1->m_gradient += m_y.m_gradient * m_x_0->m_state;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_multiply> l_element(new element_multiply(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* divide(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_divide : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_divide(

			)
			{

			}

			element_divide(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0->m_state / m_x_1->m_state;
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient / m_x_1->m_state;
				m_x_1->m_gradient += m_y.m_gradient * (-m_x_0->m_state / std::pow(m_x_1->m_state, 2.0));
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_divide> l_element(new element_divide(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* pow(
		state_gradient_pair* a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		class element_pow : public element
		{
		private:
			state_gradient_pair* m_x_0 = nullptr;
			state_gradient_pair* m_x_1 = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_pow(

			)
			{

			}

			element_pow(
				state_gradient_pair* a_x_0,
				state_gradient_pair* a_x_1
			) :
				m_x_0(a_x_0),
				m_x_1(a_x_1)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::pow(m_x_0->m_state, m_x_1->m_state);
			}

			virtual void bwd(

			)
			{
				m_x_0->m_gradient += m_y.m_gradient * m_x_1->m_state * std::pow(m_x_0->m_state, m_x_1->m_state - 1.0);
				m_x_1->m_gradient += m_y.m_gradient * std::pow(m_x_0->m_state, m_x_1->m_state) * std::log(m_x_0->m_state);
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_pow> l_element(new element_pow(a_x_0, a_x_1));
		return &l_element->m_y;
	}

	inline state_gradient_pair* sigmoid(
		state_gradient_pair* a_x
	)
	{
		class element_sigmoid : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			element_sigmoid(
				state_gradient_pair* a_x
			) :
				m_x(a_x)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = 1.0 / (1.0 + exp(-m_x->m_state));
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_y.m_gradient * m_y.m_state * (1.0 - m_y.m_state);
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_sigmoid> l_element(new element_sigmoid(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* tanh(
		state_gradient_pair* a_x
	)
	{
		class element_tanh : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_tanh(

			)
			{

			}

			element_tanh(
				state_gradient_pair* a_x
			) :
				m_x(a_x)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::tanh(m_x->m_state);
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_y.m_gradient / std::pow(cosh(m_x->m_state), 2.0);
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_tanh> l_element(new element_tanh(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* leaky_relu(
		state_gradient_pair* a_x,
		const double& a_m
	)
	{
		class element_leaky_relu : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;
			double m_m = 0;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_leaky_relu(

			)
			{

			}

			element_leaky_relu(
				state_gradient_pair* a_x,
				const double& a_m
			) :
				m_x(a_x),
				m_m(a_m)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state =
					(m_x->m_state > 0) * m_x->m_state +
					(m_x->m_state <= 0) * m_m * m_x->m_state;
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient +=
					(m_x->m_state > 0) * m_y.m_gradient +
					(m_x->m_state <= 0) * m_y.m_gradient * m_m;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_leaky_relu> l_element(new element_leaky_relu(a_x, a_m));
		return &l_element->m_y;
	}

	inline bool* branch(
		element_vector&& a_model,
		const bool& a_enabled
	)
	{
		class element_branch : public element
		{
		private:
			element_vector m_element_vector;

		public:
			bool m_enabled = false;

		public:
			virtual ~element_branch(

			)
			{

			}

			element_branch(
				element_vector&& a_element_vector,
				const bool& a_enabled
			) :
				m_element_vector(a_element_vector),
				m_enabled(a_enabled)
			{

			}

			virtual void fwd(

			)
			{
				if (m_enabled)
				{
					m_element_vector.fwd();
				}
			}

			virtual void bwd(

			)
			{
				if (m_enabled)
				{
					m_element_vector.bwd();
				}
			}

		};
		affix_base::data::ptr<element_branch> l_element(new element_branch(std::move(a_model), a_enabled));
		return &l_element->m_enabled;
	}

	inline state_gradient_pair* running_average(
		state_gradient_pair* a_x,
		double a_beta
	)
	{
		class element_running_average : public element
		{
		private:
			state_gradient_pair* m_x = nullptr;
			double m_beta = 0;
			double m_alpha = 0;

		public:
			state_gradient_pair m_y = { 1 };

		public:
			virtual ~element_running_average(

			)
			{

			}

			element_running_average(
				state_gradient_pair* a_x,
				const double& a_beta
			) :
				m_x(a_x),
				m_beta(a_beta),
				m_alpha(1.0 - a_beta)
			{
				assert(a_beta >= 0 && a_beta <= 1);
			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_beta * m_y.m_state + m_alpha * m_x->m_state;
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_alpha * m_y.m_gradient;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_running_average> l_element(new element_running_average(a_x, a_beta));
		return &l_element->m_y;
	}

	inline state_gradient_pair* log(
		state_gradient_pair* a_x
	)
	{
		class element_log
		{
		private:
			state_gradient_pair* m_x = nullptr;

		public:
			state_gradient_pair m_y;

		public:
			virtual ~element_log(

			)
			{

			}

			element_log(
				state_gradient_pair* a_x
			) :
				m_x(a_x)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::log(m_x->m_state);
			}

			virtual void bwd(

			)
			{
				m_x->m_gradient += m_y.m_gradient / m_x->m_state;
				m_y.m_gradient = 0;
			}

		};
		affix_base::data::ptr<element_log> l_element(new element_log(a_x));
		return &l_element->m_y;
	}

	inline state_gradient_pair* parameter(

	)
	{
		return parameter_vector::next();
	}

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

	inline std::vector<state_gradient_pair*> add(
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

	inline std::vector<state_gradient_pair*> subtract(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_y;
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_y.push_back(subtract(a_x_0[i], a_x_1[i]));
		}
		return l_y;
	}

	inline std::vector<state_gradient_pair*> additive_aggregate(
		std::vector<std::vector<state_gradient_pair*>> a_x
	)
	{
		std::vector<state_gradient_pair*> l_result = a_x[0];
		for (int i = 1; i < a_x.size(); i++)
			l_result = add(l_result, a_x[i]);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> transpose(
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

	inline state_gradient_pair* multiply(
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

	inline std::vector<state_gradient_pair*> multiply(
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

	inline std::vector<state_gradient_pair*> multiply(
		std::vector<std::vector<state_gradient_pair*>> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		assert(a_x_0[0].size() == a_x_1.size());
		auto l_transpose = transpose(a_x_0);
		std::vector<std::vector<state_gradient_pair*>> l_scaled_transpose;
		for (int i = 0; i < a_x_1.size(); i++)
		{
			l_scaled_transpose.push_back(multiply(l_transpose[i], a_x_1[i]));
		}
		return additive_aggregate(l_scaled_transpose);
	}

	inline std::vector<std::vector<state_gradient_pair*>> multiply(
		std::vector<std::vector<state_gradient_pair*>> a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result.push_back(multiply(a_x_0[i], a_x_1));
		}
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> multiply(
		std::vector<std::vector<state_gradient_pair*>> a_x_0,
		std::vector<std::vector<state_gradient_pair*>> a_x_1
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		std::vector<std::vector<state_gradient_pair*>> l_x_1_transpose = transpose(a_x_1);
		for (int i = 0; i < a_x_0.size(); i++)
		{
			std::vector<state_gradient_pair*> l_result_row;
			for (int j = 0; j < l_x_1_transpose.size(); j++)
			{
				l_result_row.push_back(multiply(a_x_0[i], l_x_1_transpose[j]));
			}
			l_result.push_back(l_result_row);
		}
		return l_result;
	}

	inline std::vector<state_gradient_pair*> hadamard(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_y;
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_y.push_back(multiply(a_x_0[i], a_x_1[i]));
		}
		return l_y;
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
		return multiply(parameters(a_y_size, a_x.size()), a_x);
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

	inline std::vector<std::vector<state_gradient_pair*>> sigmoid(
		std::vector<std::vector<state_gradient_pair*>> a_x
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(sigmoid(a_x[i]));
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> tanh(
		std::vector<std::vector<state_gradient_pair*>> a_x
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(tanh(a_x[i]));
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> leaky_relu(
		std::vector<std::vector<state_gradient_pair*>> a_x,
		const double& a_m
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result;
		for (int i = 0; i < a_x.size(); i++)
			l_result.push_back(leaky_relu(a_x[i], a_m));
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

	std::vector<std::vector<state_gradient_pair*>> lstm(
		std::vector<std::vector<state_gradient_pair*>> a_x,
		const size_t& a_y_size
	);

	inline state_gradient_pair* parameterized_interpolate(
		std::vector<state_gradient_pair*> a_x
	)
	{
		return multiply(
			normalize(sigmoid(parameters(a_x.size()))),
			a_x
		);
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
		auto l_multiply = multiply(a_x_0, a_x_1);
		auto l_magnitude_0 = vector_magnitude(a_x_0);
		auto l_magnitude_1 = vector_magnitude(a_x_1);
		return divide(divide(l_multiply, l_magnitude_0), l_magnitude_1);
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

	inline std::vector<state_gradient_pair*> distance_reciprocal_similarity_interpolate(
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

		auto l_transpose = transpose(a_values);

		return multiply(l_transpose, l_normalized);

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

	inline std::vector<state_gradient_pair*> concat(
		std::vector<state_gradient_pair*> a_x_0,
		std::vector<state_gradient_pair*> a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (auto& l_element : a_x_0)
			l_result.push_back(l_element);
		for (auto& l_element : a_x_1)
			l_result.push_back(l_element);
		return l_result;
	}

	inline std::vector<state_gradient_pair*> rve(
		std::vector<state_gradient_pair*> a_query,
		std::vector<state_gradient_pair*> a_key,
		std::vector<state_gradient_pair*> a_value,
		std::vector<size_t> a_hidden_dimensions = { 128 }
	)
	{
		std::vector<state_gradient_pair*> l_x = concat(a_query, a_key);
		std::vector<state_gradient_pair*> l_s = l_x;

		for (int j = 0; j < a_hidden_dimensions.size(); j++)
		{
			l_s = weight_junction(l_s, a_hidden_dimensions[j]);
			l_s = bias(l_s);
			l_s = leaky_relu(l_s, 0.3);
		}

		l_s = weight_junction(l_s, a_value.size());
		l_s = bias(l_s);
		l_s = sigmoid(l_s);

		std::vector<state_gradient_pair*> l_o = l_x;

		for (int j = 0; j < a_hidden_dimensions.size(); j++)
		{
			l_o = weight_junction(l_o, a_hidden_dimensions[j]);
			l_o = bias(l_o);
			l_o = leaky_relu(l_o, 0.3);
		}

		l_o = weight_junction(l_o, a_value.size());
		l_o = bias(l_o);
		l_o = leaky_relu(l_o, 0.3);

		return add(l_o, hadamard(l_s, a_value));

	}

	inline std::vector<state_gradient_pair*> mrve(
		std::vector<state_gradient_pair*> a_query,
		std::vector<std::vector<state_gradient_pair*>> a_keys,
		std::vector<std::vector<state_gradient_pair*>> a_values,
		std::vector<size_t> a_hidden_dimensions = { 128 }
	)
	{
		const size_t l_value_dimensions = a_values[0].size();

		std::vector<std::vector<state_gradient_pair*>> l_rve_ys;

		size_t l_rve_parameter_start_index = parameter_vector::next_index();

		for (int i = 0; i < a_keys.size(); i++)
		{
			parameter_vector::next_index(l_rve_parameter_start_index);
			l_rve_ys.push_back(rve(a_query, a_keys[i], a_values[i], a_hidden_dimensions));
		}

		std::vector<state_gradient_pair*> l_confidence_ys;

		size_t l_confidence_parameter_start_index = parameter_vector::next_index();

		for (int i = 0; i < a_keys.size(); i++)
		{
			parameter_vector::next_index(l_confidence_parameter_start_index);

			std::vector<state_gradient_pair*> l_confidence_x = concat(a_query, a_keys[i]);
			std::vector<state_gradient_pair*> l_confidence_y = l_confidence_x;

			for (int j = 0; j < a_hidden_dimensions.size(); j++)
			{
				l_confidence_y = weight_junction(l_confidence_y, a_hidden_dimensions[j]);
				l_confidence_y = bias(l_confidence_y);
				l_confidence_y = leaky_relu(l_confidence_y, 0.3);
			}

			l_confidence_y = weight_junction(l_confidence_y, 1);
			l_confidence_y = bias(l_confidence_y);
			l_confidence_y = sigmoid(l_confidence_y);

			l_confidence_ys.push_back(l_confidence_y[0]);
		}

		std::vector<state_gradient_pair*> l_normalized_confidence_ys = normalize(l_confidence_ys);

		return multiply(transpose(l_rve_ys), l_normalized_confidence_ys);

	}

	inline std::vector<state_gradient_pair*> key_vector_map(
		std::vector<state_gradient_pair*> a_x,
		const std::vector<size_t>& a_dimensions,
		const size_t& a_y_size
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_key_matrix = parameters(a_dimensions.back(), a_y_size);

		std::vector<state_gradient_pair*> l_limbic_y = a_x;

		for (int i = 0; i < a_dimensions.size(); i++)
		{
			l_limbic_y = weight_junction(l_limbic_y, a_dimensions[i]);
			l_limbic_y = bias(l_limbic_y);
			l_limbic_y = leaky_relu(l_limbic_y, 0.3);
		}

		return multiply(transpose(l_key_matrix), l_limbic_y);

	}

	inline state_gradient_pair* mean_squared_error(
		state_gradient_pair* a_prediction,
		state_gradient_pair* a_desired
	)
	{
		auto l_error = subtract(a_prediction, a_desired);
		return pow(l_error, constant(2));
	}

	inline state_gradient_pair* mean_squared_error(
		std::vector<state_gradient_pair*> a_prediction,
		std::vector<state_gradient_pair*> a_desired
	)
	{
		std::vector<state_gradient_pair*> l_squared_errors;

		for (int i = 0; i < a_prediction.size(); i++)
		{
			l_squared_errors.push_back(
				pow(
					subtract(a_prediction[i], a_desired[i]),
					constant(2))
			);
		}

		return divide(
			additive_aggregate(l_squared_errors),
			constant(a_prediction.size()));
	}

	inline state_gradient_pair* mean_squared_error(
		std::vector<std::vector<state_gradient_pair*>> a_prediction,
		std::vector<std::vector<state_gradient_pair*>> a_desired
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_squared_errors;

		for (int i = 0; i < a_prediction.size(); i++)
		{
			std::vector<state_gradient_pair*> l_squared_error_row;
			for (int j = 0; j < a_prediction[i].size(); j++)
			{
				l_squared_error_row.push_back(
					pow(
						subtract(a_prediction[i][j], a_desired[i][j]),
						constant(2))
				);
			}
			l_squared_errors.push_back(l_squared_error_row);
		}

		return divide(
			additive_aggregate(additive_aggregate(l_squared_errors)),
			constant(a_prediction.size() * a_prediction[0].size()));
	}

}
