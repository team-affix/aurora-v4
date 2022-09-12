#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"
#include "maths.h"
#include "affix-base/persistent_thread.h"

namespace aurora
{
	class element;

	struct parallel_executor
	{
	private:
		static std::vector<affix_base::threading::persistent_thread>* s_persistent_threads;

	public:
		static void startup(
			std::vector<affix_base::threading::persistent_thread>& a_persistent_threads
		)
		{
			s_persistent_threads = &a_persistent_threads;
		}

		static std::vector<affix_base::threading::persistent_thread>& threads(

		)
		{
			assert(s_persistent_threads != nullptr);
			return *s_persistent_threads;
		}

	};

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
			element_vector::insert(this);
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
			state_gradient_pair_dependency m_x_0;
			state_gradient_pair_dependency m_x_1;

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
				m_x_0(a_x_0->depend()),
				m_x_1(a_x_1->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0.m_state + m_x_1.m_state;
			}

			virtual void bwd(

			)
			{
				double l_y_gradient = m_y.gradient();
				m_x_0.m_partial_gradient = l_y_gradient;
				m_x_1.m_partial_gradient = l_y_gradient;
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
			state_gradient_pair_dependency m_x_0;
			state_gradient_pair_dependency m_x_1;

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
				m_x_0(a_x_0->depend()),
				m_x_1(a_x_1->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0.m_state - m_x_1.m_state;
			}

			virtual void bwd(

			)
			{
				double l_y_gradient = m_y.gradient();
				m_x_0.m_partial_gradient = l_y_gradient;
				m_x_1.m_partial_gradient = -l_y_gradient;
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
			state_gradient_pair_dependency m_x_0;
			state_gradient_pair_dependency m_x_1;

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
				m_x_0(a_x_0->depend()),
				m_x_1(a_x_1->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0.m_state * m_x_1.m_state;
			}

			virtual void bwd(

			)
			{
				double l_y_gradient = m_y.gradient();
				m_x_0.m_partial_gradient = l_y_gradient * m_x_1.m_state;
				m_x_1.m_partial_gradient = l_y_gradient * m_x_0.m_state;
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
			state_gradient_pair_dependency m_x_0;
			state_gradient_pair_dependency m_x_1;

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
				m_x_0(a_x_0->depend()),
				m_x_1(a_x_1->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_x_0.m_state / m_x_1.m_state;
			}

			virtual void bwd(

			)
			{
				double l_y_gradient = m_y.gradient();
				m_x_0.m_partial_gradient = l_y_gradient / m_x_1.m_state;
				m_x_1.m_partial_gradient = l_y_gradient * (-m_x_0.m_state / std::pow(m_x_1.m_state, 2.0));
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
			state_gradient_pair_dependency m_x_0;
			state_gradient_pair_dependency m_x_1;

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
				m_x_0(a_x_0->depend()),
				m_x_1(a_x_1->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::pow(m_x_0.m_state, m_x_1.m_state);
			}

			virtual void bwd(

			)
			{
				double l_y_gradient = m_y.gradient();
				m_x_0.m_partial_gradient = l_y_gradient * m_x_1.m_state * std::pow(m_x_0.m_state, m_x_1.m_state - 1.0);
				m_x_1.m_partial_gradient = l_y_gradient * std::pow(m_x_0.m_state, m_x_1.m_state) * std::log(m_x_0.m_state);
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
			state_gradient_pair_dependency m_x;

		public:
			state_gradient_pair m_y;

		public:
			element_sigmoid(
				state_gradient_pair* a_x
			) :
				m_x(a_x->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = 1.0 / (1.0 + exp(-m_x.m_state));
			}

			virtual void bwd(

			)
			{
				m_x.m_partial_gradient = m_y.gradient() * m_y.m_state * (1.0 - m_y.m_state);
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
			state_gradient_pair_dependency m_x;

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
				m_x(a_x->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::tanh(m_x.m_state);
			}

			virtual void bwd(

			)
			{
				m_x.m_partial_gradient = m_y.gradient() / std::pow(cosh(m_x.m_state), 2.0);
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
			state_gradient_pair_dependency m_x;
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
				m_x(a_x->depend()),
				m_m(a_m)
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state =
					(m_x.m_state > 0) * m_x.m_state +
					(m_x.m_state <= 0) * m_m * m_x.m_state;
			}

			virtual void bwd(

			)
			{
				double l_y_gradient = m_y.gradient();
				m_x.m_partial_gradient =
					(m_x.m_state > 0) * l_y_gradient +
					(m_x.m_state <= 0) * l_y_gradient * m_m;
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
			state_gradient_pair_dependency m_x;
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
				m_x(a_x->depend()),
				m_beta(a_beta),
				m_alpha(1.0 - a_beta)
			{
				assert(a_beta >= 0 && a_beta <= 1);
			}

			virtual void fwd(

			)
			{
				m_y.m_state = m_beta * m_y.m_state + m_alpha * m_x.m_state;
			}

			virtual void bwd(

			)
			{
				m_x.m_partial_gradient = m_alpha * m_y.gradient();
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
			state_gradient_pair_dependency m_x;

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
				m_x(a_x->depend())
			{

			}

			virtual void fwd(

			)
			{
				m_y.m_state = std::log(m_x.m_state);
			}

			virtual void bwd(

			)
			{
				m_x.m_partial_gradient = m_y.gradient() / m_x.m_state;
			}

		};
		affix_base::data::ptr<element_log> l_element(new element_log(a_x));
		return &l_element->m_y;
	}

	inline void parallel_branch(
		affix_base::threading::persistent_thread& a_persistent_thread,
		element_vector&& a_model
	)
	{
		class element_parallel_branch : public element
		{
		private:
			affix_base::threading::persistent_thread& m_persistent_thread;
			element_vector m_element_vector;

		private:
			std::function<void()> m_fwd_execute;
			std::function<void()> m_bwd_execute;

		public:
			virtual ~element_parallel_branch(

			)
			{

			}

			element_parallel_branch(
				affix_base::threading::persistent_thread& a_persistent_thread,
				element_vector&& a_element_vector
			) :
				m_persistent_thread(a_persistent_thread),
				m_element_vector(a_element_vector)
			{
				m_fwd_execute = [&]
				{
					m_element_vector.fwd();
				};
				m_bwd_execute = [&]
				{
					m_element_vector.bwd();
				};
			}

			virtual void fwd(

			)
			{
				m_persistent_thread.execute(m_fwd_execute);
			}

			virtual void bwd(

			)
			{
				m_persistent_thread.execute(m_bwd_execute);
			}

		};
		affix_base::data::ptr<element_parallel_branch> l_element(new element_parallel_branch(a_persistent_thread, std::move(a_model)));
	}
	
	inline void join_threads(

	)
	{
		class element_join_threads : public element
		{
		public:
			virtual ~element_join_threads(

			)
			{

			}

			element_join_threads(

			)
			{

			}

			virtual void fwd(

			)
			{
				for (auto& l_persistent_thread : parallel_executor::threads())
					l_persistent_thread.join();
			}

			virtual void bwd(

			)
			{
				for (auto& l_persistent_thread : parallel_executor::threads())
					l_persistent_thread.join();
			}

		};
		affix_base::data::ptr<element_join_threads> l_element(new element_join_threads());
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
		std::vector<state_gradient_pair*> l_result(a_count);
		for (int i = 0; i < a_count; i++)
			l_result[i] = parameter();
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> parameters(
		const size_t& a_rows,
		const size_t& a_cols
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_rows);
		for (int i = 0; i < a_rows; i++)
			l_result[i] = parameters(a_cols);
		return l_result;
	}

	inline std::vector<std::vector<std::vector<state_gradient_pair*>>> parameters(
		const size_t& a_depth,
		const size_t& a_rows,
		const size_t& a_cols
	)
	{
		std::vector<std::vector<std::vector<state_gradient_pair*>>> l_result(a_depth);
		for (int i = 0; i < a_depth; i++)
			l_result[i] = parameters(a_rows, a_cols);
		return l_result;
	}

	inline std::vector<state_gradient_pair*> range(
		const std::vector<state_gradient_pair*>& a_vector,
		const size_t& a_start_index,
		const size_t& a_size
	)
	{
		std::vector<state_gradient_pair*> l_result(a_size);

		for (int i = 0; i < a_size; i++)
		{
			l_result[i] = a_vector[a_start_index + i];
		}

		return l_result;

	}

	inline std::vector<std::vector<state_gradient_pair*>> range(
		const std::vector<std::vector<state_gradient_pair*>>& a_matrix,
		const size_t& a_top_index,
		const size_t& a_left_index,
		const size_t& a_height,
		const size_t& a_width
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_height);

		for (int i = 0; i < a_height; i++)
		{
			l_result[i] = range(a_matrix[a_top_index + i], a_left_index, a_width);
		}

		return l_result;

	}

	inline std::vector<std::vector<std::vector<state_gradient_pair*>>> range(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_tensor,
		const size_t& a_front_index,
		const size_t& a_top_index,
		const size_t& a_left_index,
		const size_t& a_depth,
		const size_t& a_height,
		const size_t& a_width
	)
	{
		std::vector<std::vector<std::vector<state_gradient_pair*>>> l_result(a_depth);

		for (int i = 0; i < a_depth; i++)
		{
			l_result[i] = range(a_tensor[a_front_index + i], a_top_index, a_left_index, a_height, a_width);
		}

		return l_result;

	}

	inline state_gradient_pair* additive_aggregate(
		const std::vector<state_gradient_pair*>& a_x
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

	inline state_gradient_pair* additive_aggregate_parallel(
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		assert(a_x.size() > 0);

		std::vector<affix_base::threading::persistent_thread>& l_persistent_threads = parallel_executor::threads();

		std::vector<std::vector<state_gradient_pair*>> l_x_partitioned(l_persistent_threads.size());

		for (int i = 0; i < a_x.size(); i++)
		{
			size_t l_persistent_thread_index = i % l_persistent_threads.size();
			l_x_partitioned[l_persistent_thread_index].push_back(a_x[i]);
		}

		std::vector<state_gradient_pair*> l_y_partitioned(l_persistent_threads.size());

		join_threads();

		for (int i = 0; i < l_persistent_threads.size(); i++)
		{
			element_vector::start();

			l_y_partitioned[i] = additive_aggregate(l_x_partitioned[i]);

			parallel_branch(l_persistent_threads[i], element_vector::stop());

		}

		join_threads();

		return additive_aggregate(l_y_partitioned);

	}

	inline std::vector<state_gradient_pair*> add(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());
		std::vector<state_gradient_pair*> l_result(a_x_0.size());
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result[i] = add(a_x_0[i], a_x_1[i]);
		}
		return l_result;
	}

	inline std::vector<state_gradient_pair*> subtract(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_y(a_x_0.size());
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_y[i] = subtract(a_x_0[i], a_x_1[i]);
		}
		return l_y;
	}

	inline std::vector<state_gradient_pair*> additive_aggregate(
		const std::vector<std::vector<state_gradient_pair*>>& a_x
	)
	{
		std::vector<state_gradient_pair*> l_result = a_x[0];
		for (int i = 1; i < a_x.size(); i++)
			l_result = add(l_result, a_x[i]);
		return l_result;
	}

	inline state_gradient_pair* average(
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		return divide(
			additive_aggregate(a_x),
			constant(a_x.size()));
	}

	inline std::vector<std::vector<state_gradient_pair*>> transpose(
		const std::vector<std::vector<state_gradient_pair*>>& a_x
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

	inline std::vector<state_gradient_pair*> hadamard(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<state_gradient_pair*> l_y(a_x_0.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_y[i] = multiply(a_x_0[i], a_x_1[i]);
		}

		return l_y;

	}

	inline std::vector<std::vector<state_gradient_pair*>> hadamard(
		const std::vector<std::vector<state_gradient_pair*>>& a_x_0,
		const std::vector<std::vector<state_gradient_pair*>>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<std::vector<state_gradient_pair*>> l_y(a_x_0.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_y[i] = hadamard(a_x_0[i], a_x_1[i]);
		}

		return l_y;

	}

	inline std::vector<std::vector<std::vector<state_gradient_pair*>>> hadamard(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_x_0,
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<std::vector<std::vector<state_gradient_pair*>>> l_y(a_x_0.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_y[i] = hadamard(a_x_0[i], a_x_1[i]);
		}

		return l_y;

	}

	inline std::vector<state_gradient_pair*> hadamard_parallel(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<affix_base::threading::persistent_thread>& l_persistent_threads = parallel_executor::threads();

		std::vector<std::vector<state_gradient_pair*>> l_multiply_y(l_persistent_threads.size());
		std::vector<std::vector<state_gradient_pair*>> l_multiply_x0(l_persistent_threads.size());
		std::vector<std::vector<state_gradient_pair*>> l_multiply_x1(l_persistent_threads.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			size_t l_thread_index = i % l_persistent_threads.size();
			l_multiply_y[l_thread_index].push_back(0);
			l_multiply_x0[l_thread_index].push_back(a_x_0[i]);
			l_multiply_x1[l_thread_index].push_back(a_x_1[i]);
		}

		join_threads();

		for (int i = 0; i < l_persistent_threads.size(); i++)
		{
			auto& l_y_partition = l_multiply_y[i];
			auto& l_x0_partition = l_multiply_x0[i];
			auto& l_x1_partition = l_multiply_x1[i];
			element_vector::start();
			for (int j = 0; j < l_y_partition.size(); j++)
			{
				l_y_partition[j] = multiply(l_x0_partition[j], l_x1_partition[j]);
			}
			parallel_branch(l_persistent_threads[i], element_vector::stop());
		}

		join_threads();

		std::vector<state_gradient_pair*> l_hadamard_y(a_x_0.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			size_t l_thread_index = i % l_persistent_threads.size();
			size_t l_partition_element_index = i / l_persistent_threads.size();
			l_hadamard_y[i] = l_multiply_y[l_thread_index][l_partition_element_index];
		}

		return l_hadamard_y;

	}

	inline state_gradient_pair* multiply(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<state_gradient_pair*> l_multiply_ys(a_x_0.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_multiply_ys[i] = multiply(a_x_0[i], a_x_1[i]);
		}

		return additive_aggregate(l_multiply_ys);

	}

	inline state_gradient_pair* multiply_parallel(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<state_gradient_pair*> l_hadamard_y = hadamard_parallel(a_x_0, a_x_1);

		return additive_aggregate_parallel(l_hadamard_y);

	}

	inline std::vector<state_gradient_pair*> multiply(
		const std::vector<state_gradient_pair*>& a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x_0.size());
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result[i] = multiply(a_x_0[i], a_x_1);
		}
		return l_result;
	}

	inline std::vector<state_gradient_pair*> multiply(
		const std::vector<std::vector<state_gradient_pair*>>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0[0].size() == a_x_1.size());
		auto l_transpose = transpose(a_x_0);
		std::vector<std::vector<state_gradient_pair*>> l_scaled_transpose(l_transpose.size());
		for (int i = 0; i < a_x_1.size(); i++)
		{
			l_scaled_transpose[i] = multiply(l_transpose[i], a_x_1[i]);
		}
		return additive_aggregate(l_scaled_transpose);
	}

	inline std::vector<std::vector<state_gradient_pair*>> multiply(
		const std::vector<std::vector<state_gradient_pair*>>& a_x_0,
		state_gradient_pair* a_x_1
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_x_0.size());
		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result[i] = multiply(a_x_0[i], a_x_1);
		}
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> multiply(
		const std::vector<std::vector<state_gradient_pair*>>& a_x_0,
		const std::vector<std::vector<state_gradient_pair*>>& a_x_1
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_x_0.size());
		std::vector<std::vector<state_gradient_pair*>> l_x_1_transpose = transpose(a_x_1);
		for (int i = 0; i < a_x_0.size(); i++)
		{
			std::vector<state_gradient_pair*> l_result_row(a_x_1[0].size());
			for (int j = 0; j < l_x_1_transpose.size(); j++)
			{
				l_result_row[j] = multiply(a_x_0[i], l_x_1_transpose[j]);
			}
			l_result[i] = l_result_row;
		}
		return l_result;
	}

	inline state_gradient_pair* bias(
		state_gradient_pair* a_x
	)
	{
		return add(parameter(), a_x);
	}

	inline std::vector<state_gradient_pair*> bias(
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x.size());

		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = bias(a_x[i]);

		return l_result;

	}

	inline std::vector<std::vector<state_gradient_pair*>> bias(
		const std::vector<std::vector<state_gradient_pair*>>& a_x
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_x.size());

		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = bias(a_x[i]);

		return l_result;

	}

	inline std::vector<std::vector<std::vector<state_gradient_pair*>>> bias(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_x
	)
	{
		std::vector<std::vector<std::vector<state_gradient_pair*>>> l_result(a_x.size());

		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = bias(a_x[i]);

		return l_result;

	}

	inline std::vector<state_gradient_pair*> weight_junction(
		const std::vector<state_gradient_pair*>& a_x,
		const size_t& a_y_size
	)
	{
		return multiply(parameters(a_y_size, a_x.size()), a_x);
	}

	inline std::vector<state_gradient_pair*> sigmoid(
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = sigmoid(a_x[i]);
		return l_result;
	}

	inline std::vector<state_gradient_pair*> tanh(
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = tanh(a_x[i]);
		return l_result;
	}

	inline std::vector<state_gradient_pair*> leaky_relu(
		const std::vector<state_gradient_pair*>& a_x,
		const double& a_m
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = leaky_relu(a_x[i], a_m);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> sigmoid(
		const std::vector<std::vector<state_gradient_pair*>>& a_x
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = sigmoid(a_x[i]);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> tanh(
		const std::vector<std::vector<state_gradient_pair*>>& a_x
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = tanh(a_x[i]);
		return l_result;
	}

	inline std::vector<std::vector<state_gradient_pair*>> leaky_relu(
		const std::vector<std::vector<state_gradient_pair*>>& a_x,
		const double& a_m
	)
	{
		std::vector<std::vector<state_gradient_pair*>> l_result(a_x.size());
		for (int i = 0; i < a_x.size(); i++)
			l_result[i] = leaky_relu(a_x[i], a_m);
		return l_result;
	}

	inline state_gradient_pair* multiplicative_aggregate(
		const std::vector<state_gradient_pair*>& a_x
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
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x.size());

		state_gradient_pair* l_sum = additive_aggregate(a_x);

		for (int i = 0; i < a_x.size(); i++)
		{
			l_result[i] = divide(a_x[i], l_sum);
		}

		return l_result;

	}

	std::vector<std::vector<state_gradient_pair*>> lstm(
		const std::vector<std::vector<state_gradient_pair*>>& a_x,
		const size_t& a_y_size
	);

	inline state_gradient_pair* parameterized_interpolate(
		const std::vector<state_gradient_pair*>& a_x
	)
	{
		return multiply(
			normalize(sigmoid(parameters(a_x.size()))),
			a_x
		);
	}

	inline state_gradient_pair* vector_magnitude(
		const std::vector<state_gradient_pair*>& a_x
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
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		auto l_multiply = multiply(a_x_0, a_x_1);
		auto l_magnitude_0 = vector_magnitude(a_x_0);
		auto l_magnitude_1 = vector_magnitude(a_x_1);
		return divide(divide(l_multiply, l_magnitude_0), l_magnitude_1);
	}

	inline std::vector<state_gradient_pair*> vector_vector_subtract(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		assert(a_x_0.size() == a_x_1.size());

		std::vector<state_gradient_pair*> l_result(a_x_0.size());

		for (int i = 0; i < a_x_0.size(); i++)
		{
			l_result[i] = subtract(a_x_0[i], a_x_1[i]);
		}

		return l_result;

	}

	inline state_gradient_pair* euclidian_distance(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		return vector_magnitude(vector_vector_subtract(a_x_0, a_x_1));
	}

	inline std::vector<state_gradient_pair*> similarity_interpolate(
		const std::vector<state_gradient_pair*>& a_query,
		const std::vector<std::vector<state_gradient_pair*>>& a_keys,
		const std::vector<std::vector<state_gradient_pair*>>& a_values
	)
	{
		std::vector<state_gradient_pair*> l_similarity_ys(a_keys.size()); // Each element between 0 and +inf

		for (int i = 0; i < a_keys.size(); i++)
		{
			auto l_distance = euclidian_distance(a_query, a_keys[i]);
			auto l_stabilized = add(l_distance, constant(0.0000001));
			l_similarity_ys[i] = divide(constant(1), l_stabilized);
		}

		auto l_normalized = normalize(l_similarity_ys);

		auto l_transpose = transpose(a_values);

		return multiply(l_transpose, l_normalized);

	}

	inline std::vector<std::vector<state_gradient_pair*>> partition(
		const std::vector<state_gradient_pair*>& a_x,
		const size_t& a_bin_size
	)
	{
		assert(a_x.size() % a_bin_size == 0);

		size_t l_bins_count = a_x.size() / a_bin_size;

		std::vector<std::vector<state_gradient_pair*>> l_result(l_bins_count);

		for (int i = 0; i < l_bins_count; i++)
		{
			std::vector<state_gradient_pair*> l_bin(a_bin_size);

			for (int j = 0; j < a_bin_size; j++)
				l_bin[j] = a_x[a_bin_size * i + j];

			l_result[i] = l_bin;

		}

		return l_result;

	}

	inline std::vector<state_gradient_pair*> concat(
		const std::vector<state_gradient_pair*>& a_x_0,
		const std::vector<state_gradient_pair*>& a_x_1
	)
	{
		std::vector<state_gradient_pair*> l_result(a_x_0.size() + a_x_1.size());
		for (int i = 0; i < a_x_0.size(); i++)
			l_result[i] = a_x_0[i];
		for (int i = 0; i < a_x_1.size(); i++)
			l_result[a_x_0.size() + i] = a_x_1[i];
		return l_result;
	}

	inline std::vector<state_gradient_pair*> flatten(
		const std::vector<std::vector<state_gradient_pair*>>& a_matrix
	)
	{
		std::vector<state_gradient_pair*> l_result(a_matrix.size() * a_matrix[0].size());

		for (int i = 0; i < a_matrix.size(); i++)
		{
			for (int j = 0; j < a_matrix[0].size(); j++)
			{
				l_result[i * a_matrix[0].size() + j] = a_matrix[i][j];
			}
		}

		return l_result;

	}

	inline std::vector<state_gradient_pair*> flatten(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_tensor
	)
	{
		std::vector<state_gradient_pair*> l_result;
		for (int i = 0; i < a_tensor.size(); i++)
		{
			std::vector<state_gradient_pair*> l_flattened_matrix = flatten(a_tensor[i]);
			l_result.insert(l_result.end(), l_flattened_matrix.begin(), l_flattened_matrix.end());
		}
		return l_result;
	}

	inline std::vector<state_gradient_pair*> convolve(
		const std::vector<std::vector<state_gradient_pair*>>& a_x,
		const std::vector<std::vector<state_gradient_pair*>>& a_filter,
		const size_t& a_stride = 1
	)
	{
		// Since the first dimension is considered to be height of the filter, we reserve the first dimension as
		// being non-spacial, and hence we use only the [0].size() as the width of our matrix.

		int l_right_most_position = a_x[0].size() - a_filter[0].size();

		assert(l_right_most_position >= 0);

		size_t l_convolution_count = (l_right_most_position / a_stride) + 1;

		std::vector<state_gradient_pair*> l_result(l_convolution_count);

		for (int i = 0; i < l_convolution_count; i++)
		{
			l_result[i] = multiply(
					flatten(
						a_filter
					),
					flatten(
						range(a_x, 0, i * a_stride, a_filter.size(), a_filter[0].size())
					)
			);
		}

		return l_result;

	}

	inline std::vector<std::vector<state_gradient_pair*>> convolve(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_x,
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_filter,
		const size_t& a_stride = 1
	)
	{
		// Since the first dimension is considered to be depth of the filter, we reserve the first dimension as
		// being non-spacial, and hence we use only the [0].size() and [0][0].size() as the 
		// height and width of our matrices, respectively.

		int l_bottom_most_position = a_x[0].size() - a_filter[0].size();
		int l_right_most_position = a_x[0][0].size() - a_filter[0][0].size();

		assert(l_bottom_most_position >= 0 && l_right_most_position >= 0);

		size_t l_vertical_convolution_count = (l_bottom_most_position / a_stride) + 1;
		size_t l_horizontal_convolution_count = (l_right_most_position / a_stride) + 1;

		std::vector<std::vector<state_gradient_pair*>> l_result(l_vertical_convolution_count);

		for (int i = 0; i < l_vertical_convolution_count; i++)
		{
			std::vector<state_gradient_pair*> l_result_row(l_horizontal_convolution_count);
			for (int j = 0; j < l_horizontal_convolution_count; j++)
			{
				l_result_row[j] = multiply(
						flatten(
							a_filter
						),
						flatten(
							range(a_x, 0, i * a_stride, j * a_stride, a_filter.size(), a_filter[0].size(), a_filter[0][0].size())
						)
				);
			}
			l_result[i] = l_result_row;
		}

		return l_result;

	}

	inline std::vector<state_gradient_pair*> average_pool(
		const std::vector<state_gradient_pair*>& a_x,
		const size_t& a_bin_width,
		const size_t& a_stride = 1
	)
	{
		size_t l_right_most_index = a_x.size() - a_bin_width;

		size_t l_pool_size = (l_right_most_index / a_stride) + 1;

		std::vector<state_gradient_pair*> l_result(l_pool_size);

		for (int i = 0; i < l_pool_size; i++)
		{
			l_result[i] = average(range(a_x, i * a_stride, a_bin_width));
		}

		return l_result;

	}

	inline std::vector<std::vector<state_gradient_pair*>> average_pool(
		const std::vector<std::vector<state_gradient_pair*>>& a_x,
		const size_t& a_bin_height,
		const size_t& a_bin_width,
		const size_t& a_stride = 1
	)
	{
		size_t l_top_most_index = a_x.size() - a_bin_height;
		size_t l_right_most_index = a_x[0].size() - a_bin_width;

		size_t l_pool_height = (l_top_most_index / a_stride) + 1;
		size_t l_pool_width = (l_right_most_index / a_stride) + 1;

		std::vector<std::vector<state_gradient_pair*>> l_result(l_pool_height);

		for (int i = 0; i < l_pool_height; i++)
		{
			std::vector<state_gradient_pair*> l_result_row(l_pool_width);

			for (int j = 0; j < l_pool_width; j++)
			{
				l_result_row[j] = average(flatten(range(a_x, i * a_stride, j * a_stride, a_bin_height, a_bin_width)));
			}

			l_result[i] = l_result_row;

		}

		return l_result;

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
		const std::vector<state_gradient_pair*>& a_prediction,
		const std::vector<state_gradient_pair*>& a_desired
	)
	{
		assert(a_prediction.size() == a_desired.size());

		std::vector<state_gradient_pair*> l_squared_errors(a_prediction.size());

		for (int i = 0; i < a_prediction.size(); i++)
		{
			l_squared_errors[i] =
				pow(
					subtract(a_prediction[i], a_desired[i]),
					constant(2)
				);
		}

		return divide(
			additive_aggregate(l_squared_errors),
			constant(a_prediction.size()));
	}

	inline state_gradient_pair* mean_squared_error(
		const std::vector<std::vector<state_gradient_pair*>>& a_prediction,
		const std::vector<std::vector<state_gradient_pair*>>& a_desired
	)
	{
		assert(a_prediction.size() == a_desired.size());
		assert(a_prediction[0].size() == a_desired[0].size());

		std::vector<std::vector<state_gradient_pair*>> l_squared_errors(a_prediction.size());

		for (int i = 0; i < a_prediction.size(); i++)
		{
			std::vector<state_gradient_pair*> l_squared_error_row(a_prediction[0].size());
			for (int j = 0; j < a_prediction[i].size(); j++)
			{
				l_squared_error_row[j] =
					pow(
						subtract(a_prediction[i][j], a_desired[i][j]),
						constant(2)
					);
			}
			l_squared_errors[i] = l_squared_error_row;
		}

		return divide(
			additive_aggregate(additive_aggregate(l_squared_errors)),
			constant(a_prediction.size() * a_prediction[0].size()));
	}

	inline state_gradient_pair* mean_squared_error(
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_prediction,
		const std::vector<std::vector<std::vector<state_gradient_pair*>>>& a_desired
	)
	{
		assert(a_prediction.size() == a_desired.size());
		assert(a_prediction[0].size() == a_desired[0].size());
		assert(a_prediction[0][0].size() == a_desired[0][0].size());

		std::vector<state_gradient_pair*> l_squared_errors(a_prediction.size() * a_prediction[0].size() * a_prediction[0][0].size());

		for (int i = 0; i < a_prediction.size(); i++)
		{
			for (int j = 0; j < a_prediction[0].size(); j++)
			{
				for (int k = 0; k < a_prediction[0][0].size(); k++)
				{
					l_squared_errors[i * a_prediction[0].size() + j * a_prediction[0][0].size() + k] =
						pow(subtract(a_prediction[i][j][k], a_desired[i][j][k]), constant(2));
				}
			}
		}

		return divide(
			additive_aggregate(l_squared_errors),
			constant(a_prediction.size() * a_prediction[0].size() * a_prediction[0][0].size()));
	}

}
