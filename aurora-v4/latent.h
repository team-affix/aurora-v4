#ifndef LATENT_H
#define LATENT_H

#include <vector>
#include <memory>
#include <assert.h>
#include <stdexcept>
#include <functional>
#include "fundamentals.h"

namespace aurora
{
	namespace latent
	{
		struct state_gradient_pair_dependency
		{
			double& m_state;
			double& m_partial_gradient;
		};

		struct state_gradient_pair
		{
			double m_state = 0;

			std::vector<std::shared_ptr<double>> m_partial_gradients;

			state_gradient_pair(

			)
			{

			}

			state_gradient_pair(
				const double& a_state
			) :
				m_state(a_state)
			{

			}

			double gradient(

			) const
			{
				double l_result = 0;
				for (const auto& l_partial_gradient : m_partial_gradients)
				{
					l_result += *l_partial_gradient;
				}
				return l_result;
			}

			state_gradient_pair_dependency depend(

			)
			{
				m_partial_gradients.push_back(std::shared_ptr<double>(new double(0)));
				return state_gradient_pair_dependency{ m_state, *m_partial_gradients.back() };
			}

		};

        template<typename T>
        inline T* pointers(
            T& a_x
        )
        {
            return &a_x;
        }

        template<typename T, size_t I, size_t ... J>
        inline tensor<T*, I, J ...> pointers(
            tensor<T, I, J ...>& a_tensor
        )
        {
            tensor<T*, I, J...> l_result;

            for (int i = 0; i < I; i++)
                l_result[i] = pointers(a_tensor[i]);

            return l_result;

        }

        inline double get_state(
            const state_gradient_pair* a_sgp_ptr
        )
        {
            return a_sgp_ptr->m_state;
        }

        template<size_t I, size_t ... J>
        inline tensor<double, I, J ...> get_state(
            const tensor<state_gradient_pair*, I, J ...>& a_tensor
        )
        {
            tensor<double, I, J ...> l_result;

            for (int i = 0; i < I; i++)
                l_result[i] = get_state(a_tensor[i]);

            return l_result;

        }

        inline void set_state(
            state_gradient_pair* a_destination,
            const double& a_source
        )
        {
            a_destination->m_state = a_source;
        }

        template<size_t I, size_t ... J>
        inline void set_state(
            tensor<state_gradient_pair*, I, J ...>& a_destination,
            const tensor<double, I, J ...>& a_source
        )
        {
            for (int i = 0; i < I; i++)
                set_state(a_destination[i], a_source[i]);
        }

        inline double get_gradient(
            const state_gradient_pair* a_sgp_ptr
        )
        {
            return a_sgp_ptr->gradient();
        }

        template<size_t I, size_t ... J>
        inline tensor<double, I, J ...> get_gradient(
            const tensor<state_gradient_pair*, I, J ...>& a_tensor
        )
        {
            tensor<double, I, J ...> l_result;

            for (int i = 0; i < I; i++)
                l_result[i] = get_gradient(a_tensor[i]);

            return l_result;

        }

        /// @brief 
        /// @tparam T is the value type.
        /// @tparam B is the number of bins into which the input should be partitioned.
        /// @tparam I is the outermost rank size.
        /// @tparam ...J the remaining rank sizes.
        /// @param a_x 
        /// @return 
        template<size_t B, typename T, size_t I, size_t ... J>
            requires ((I % B) == 0)
        tensor<T, B, I/B, J ...> partition(
            const tensor<T, I, J ...>& a_x
        )
        {
            constexpr size_t BIN_SIZE = I/B;

            tensor<T, B, BIN_SIZE, J ...> l_result;

            for (int i = 0; i < I; i++)
            {
                l_result[i / BIN_SIZE][i % BIN_SIZE] = a_x[i];
            }

            return l_result;

        }

        template<typename T, size_t I1, size_t I2, size_t ... J>
        tensor<T, I1+I2, J ...> concat(
            const tensor<T, I1, J ...>& a_x_0,
            const tensor<T, I2, J ...>& a_x_1
        )
        {
            tensor<T, I1+I2, J ...> l_result;
            
            for (int i = 0; i < I1; i++)
                l_result[i] = a_x_0[i];
            
            for (int i = 0; i < I2; i++)
                l_result[I1 + i] = a_x_1[i];

            return l_result;

        }

        template<typename T, size_t I, size_t J>
        tensor<T, I * J> flatten(
            const tensor<T, I, J>& a_tensor
        )
        {
            tensor<T, I * J> l_result;

            for (int i = 0; i < I; i++)
            {
                std::copy(a_tensor[i].begin(), a_tensor[i].end(), l_result.begin() + i * J);
            }

            return l_result;

        }

        template<typename T, size_t I, size_t ... J>
        tensor<T, (I * ... * J)> flatten(
            const tensor<T, I, J ...>& a_tensor
        )
        {
            tensor<T, (I * ... * J)> l_result;

            for (int i = 0; i < I; i++)
            {
                tensor<T, (1 * ... * J)> l_flattened_nested = flatten(a_tensor[i]);
                std::copy(l_flattened_nested.begin(), l_flattened_nested.end(), l_result.begin() + i * (1 * ... * J));
            }

            return l_result;

        }

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

        class model
        {
        private:
            std::vector<element*>             m_elements;
            std::vector<state_gradient_pair*> m_parameters;
            size_t                            m_next_index = 0;
            std::function<double()>           m_initialize_parameter;

        public:
            virtual ~model(

            )
            {
                for (auto& l_element : m_elements)
                    delete l_element;
                for (auto& l_parameter : m_parameters)
                    delete l_parameter;
            }

            model(
                const std::function<double()>& a_initialize_parameter = 
                    []
                    {
                        static std::uniform_real_distribution<double> s_urd(-1, 1);
                        return s_urd(i_default_random_engine);
                    }
            ) :
                m_initialize_parameter(a_initialize_parameter)
            {

            }
        
			void next_parameter_index(
				const size_t& a_next_index
			)
			{
				if (a_next_index > m_parameters.size())
					throw std::runtime_error("Error: a_parameter_index was out of legal bounds given s_parameter_vector's size.");
				m_next_index = a_next_index;
			}

			size_t next_parameter_index(

			)
			{
				return m_next_index;
			}

			state_gradient_pair* parameter(

			)
			{
				if (m_next_index == m_parameters.size())
				{
					m_parameters.push_back(new state_gradient_pair(m_initialize_parameter()));
					m_next_index++;
					return m_parameters.back();
				}
				else
				{
					state_gradient_pair* l_result = m_parameters.at(m_next_index);
					m_next_index++;
					return l_result;
				}
			}

            void fwd(

            )
            {
                for (int i = 0; i < m_elements.size(); i++)
                    m_elements[i]->fwd();
            }

            void bwd(

            )
            {
                for (int i = m_elements.size() - 1; i >= 0; i--)
                    m_elements[i]->bwd();
            }

            const std::vector<element*>& elements(

            ) const
            {
                return m_elements;
            }

            const std::vector<state_gradient_pair*>& parameters(

            ) const
            {
                return m_parameters;
            }

        public:
        
            state_gradient_pair* constant(
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

                element_constant* l_element(new element_constant(a_state));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;

            }

            state_gradient_pair* add(
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
                
                element_add* l_element(new element_add(a_x_0, a_x_1));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;
                
            }

            state_gradient_pair* subtract(
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

                element_subtract* l_element(new element_subtract(a_x_0, a_x_1));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;
                
            }

            state_gradient_pair* multiply(
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

                element_multiply* l_element(new element_multiply(a_x_0, a_x_1));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;

            }

            state_gradient_pair* divide(
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
                
                element_divide* l_element(new element_divide(a_x_0, a_x_1));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;
                
            }

            state_gradient_pair* pow(
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
                
                element_pow* l_element(new element_pow(a_x_0, a_x_1));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;

            }

            state_gradient_pair* sigmoid(
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
                
                element_sigmoid* l_element(new element_sigmoid(a_x));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;
                
            }

            state_gradient_pair* tanh(
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

                element_tanh* l_element(new element_tanh(a_x));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;
                
            }

            state_gradient_pair* leaky_relu(
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

                element_leaky_relu* l_element(new element_leaky_relu(a_x, a_m));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;

            }

            state_gradient_pair* log(
                state_gradient_pair* a_x
            )
            {
                class element_log : public element
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
                
                element_log* l_element(new element_log(a_x));
                
                m_elements.push_back(l_element);

                return &l_element->m_y;
                
            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, I, J ...> parameter(
                
            )
            {
                tensor<state_gradient_pair*, I, J ...> l_result;
                
                for (int i = 0; i < a_count; i++)
                    l_result[i] = parameter<J ...>();

                return l_result;

            }

            template<size_t I>
            state_gradient_pair* additive_aggregate(
                const tensor<state_gradient_pair*, I>& a_x
            )
            {
                state_gradient_pair* l_result = constant(0);

                for (int i = 0; i < I; i++)
                    l_result = add(l_result, a_x[i]);

                return l_result;

            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, J ...> additive_aggregate(
                const tensor<state_gradient_pair*, I, J ...>& a_tensor
            )
            {
                tensor<state_gradient_pair*, J ...> l_result;

                for (int i = 0; i < I; i++)
                    l_result = add(l_result, a_tensor[i]);

                return l_result;

            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, I, J ...> add(
                const tensor<state_gradient_pair*, I, J ...>& a_x_0,
                const tensor<state_gradient_pair*, I, J ...>& a_x_1
            )
            {
                tensor<state_gradient_pair*, I, J ...> l_result;

                for (int i = 0; i < I; i++)
                    l_result[i] = add(a_x_0[i], a_x_1[i]);

                return l_result;

            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, I, J ...> subtract(
                const tensor<state_gradient_pair*, I, J ...>& a_x_0,
                const tensor<state_gradient_pair*, I, J ...>& a_x_1
            )
            {
                tensor<state_gradient_pair*, I, J ...> l_result;

                for (int i = 0; i < I; i++)
                    l_result[i] = subtract(a_x_0[i], a_x_1[i]);

                return l_result;

            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, I, J ...> multiply(
                const tensor<state_gradient_pair*, I, J ...>& a_x_0,
                const state_gradient_pair* a_x_1
            )
            {
                tensor<state_gradient_pair*, I, J ...> l_result;

                for (int i = 0; i < I; i++)
                    l_result[i] = multiply(a_x_0[i], a_x_1);

                return l_result;

            }

            template<size_t I>
            state_gradient_pair* average(
                const tensor<state_gradient_pair*, I>& a_x
            )
            {
                return divide(
                    additive_aggregate(a_x),
                    constant(I));
            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, J ...> average(
                const tensor<state_gradient_pair*, I, J ...>& a_x
            )
            {
                return multiply(additive_aggregate(a_x), constant(1.0 / I));
            }

            template<size_t I, size_t J>
            tensor<state_gradient_pair*, J, I> transpose(
                const tensor<state_gradient_pair*, I, J>& a_tensor
            )
            {
                tensor<state_gradient_pair*, J, I> l_result;

                for (int i = 0; i < I; i++)
                {
                    for (int j = 0; j < J; j++)
                    {
                        l_result[j][i] = a_tensor[i][j];
                    }
                }

                return l_result;

            }

            /// @brief This function currently flips the tensor's final two ranks.
            /// @tparam I 
            /// @tparam J 
            /// @tparam K 
            /// @tparam ...L 
            /// @param a_tensor 
            /// @return 
            template<size_t I, size_t J, size_t K, size_t ... L>
            tensor<state_gradient_pair*, I, L ..., K, J> transpose(
                const tensor<state_gradient_pair*, I, L ..., J, K>& a_tensor
            )
            {
                tensor<state_gradient_pair*, I, L ..., K, J> l_result;

                for (int i = 0; i < I; i++)
                    l_result[i] = transpose(a_tensor[i]);

                return l_result;

            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, I, J ...> negate(
                const tensor<state_gradient_pair*, I, J ...>& a_tensor
            )
            {
                return multiply(a_tensor, constant(-1.0));
            }

            state_gradient_pair* bias(
                state_gradient_pair* a_x
            )
            {
                return add(parameter(), a_x);
            }

            template<size_t I, size_t ... J>
            tensor<state_gradient_pair*, I, J ...> bias(
                const tensor<state_gradient_pair*, I, J ...>& a_x
            )
            {
                tensor<state_gradient_pair*, I, J ...> l_result;

                for (int i = 0; i < I; i++)
                    l_result[i] = bias(a_x[i]);

                return l_result;
                
            }

            sgp_ptr_vector weight_junction(
                const sgp_ptr_vector& a_x,
                const size_t& a_y_size
            )
            {
                return multiply(parameters(a_y_size, a_x.size()), a_x);
            }

            sgp_ptr_vector sigmoid(
                const sgp_ptr_vector& a_x
            )
            {
                sgp_ptr_vector l_result(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                    l_result[i] = sigmoid(a_x[i]);
                return l_result;
            }

            sgp_ptr_vector tanh(
                const sgp_ptr_vector& a_x
            )
            {
                sgp_ptr_vector l_result(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                    l_result[i] = tanh(a_x[i]);
                return l_result;
            }

            sgp_ptr_vector leaky_relu(
                const sgp_ptr_vector& a_x,
                const double& a_m
            )
            {
                sgp_ptr_vector l_result(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                    l_result[i] = leaky_relu(a_x[i], a_m);
                return l_result;
            }

            sgp_ptr_matrix sigmoid(
                const sgp_ptr_matrix& a_x
            )
            {
                sgp_ptr_matrix l_result(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                    l_result[i] = sigmoid(a_x[i]);
                return l_result;
            }

            sgp_ptr_matrix tanh(
                const sgp_ptr_matrix& a_x
            )
            {
                sgp_ptr_matrix l_result(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                    l_result[i] = tanh(a_x[i]);
                return l_result;
            }

            sgp_ptr_matrix leaky_relu(
                const sgp_ptr_matrix& a_x,
                const double& a_m
            )
            {
                sgp_ptr_matrix l_result(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                    l_result[i] = leaky_relu(a_x[i], a_m);
                return l_result;
            }

            sgp_ptr_vector normalize(
                const sgp_ptr_vector& a_x
            )
            {
                sgp_ptr_vector l_result(a_x.size());

                state_gradient_pair* l_sum = additive_aggregate(a_x);

                for (int i = 0; i < a_x.size(); i++)
                {
                    l_result[i] = divide(a_x[i], l_sum);
                }

                return l_result;

            }

            sgp_ptr_matrix lstm(
                const sgp_ptr_matrix& a_x,
                const size_t& a_y_size
            )
            {
                struct lstm_timestep
                {
                public:
                    sgp_ptr_vector m_cy;
                    sgp_ptr_vector m_y;

                public:
                    lstm_timestep(
                        model& a_model,
                        sgp_ptr_vector a_x,
                        sgp_ptr_vector a_cx,
                        sgp_ptr_vector a_hx
                    )
                    {
                        sgp_ptr_vector l_hx_x_concat = concat(a_hx, a_x);

                        // Construct gates

                        auto l_forget_gate = a_model.weight_junction(l_hx_x_concat, a_hx.size());
                        l_forget_gate = a_model.bias(l_forget_gate);
                        l_forget_gate = a_model.sigmoid(l_forget_gate);

                        auto l_input_limit_gate = a_model.weight_junction(l_hx_x_concat, a_hx.size());
                        l_input_limit_gate = a_model.bias(l_input_limit_gate);
                        l_input_limit_gate = a_model.sigmoid(l_input_limit_gate);

                        auto l_input_gate = a_model.weight_junction(l_hx_x_concat, a_hx.size());
                        l_input_gate = a_model.bias(l_input_gate);
                        l_input_gate = a_model.tanh(l_input_gate);

                        auto l_output_gate = a_model.weight_junction(l_hx_x_concat, a_hx.size());
                        l_output_gate = a_model.bias(l_output_gate);
                        l_output_gate = a_model.sigmoid(l_output_gate);


                        // Forget parts of the cell state
                        sgp_ptr_vector l_cell_state_after_forget = a_model.hadamard(a_cx, l_forget_gate);

                        // Calculate the input to the cell state
                        sgp_ptr_vector l_limited_input = a_model.hadamard(l_input_gate, l_input_limit_gate);

                        // Write the input to the cell state
                        sgp_ptr_vector l_cell_state_after_input = a_model.add(l_cell_state_after_forget, l_limited_input);

                        // Cell state is now finalized, save it as the cell state output
                        m_cy = l_cell_state_after_input;

                        // Do a temporary step to compute tanh(cy)
                        sgp_ptr_vector l_cell_state_after_tanh = a_model.tanh(l_cell_state_after_input);

                        // Compute output to the timestep
                        m_y = a_model.hadamard(l_output_gate, l_cell_state_after_tanh);

                    }

                };

                sgp_ptr_matrix l_result(a_x.size());

                sgp_ptr_vector l_cy = parameters(a_y_size);
                sgp_ptr_vector l_hy = parameters(a_y_size);

                size_t l_timestep_parameters_start_index = next_parameter_index();

                for (int i = 0; i < a_x.size(); i++)
                {
                    next_parameter_index(l_timestep_parameters_start_index);
                    lstm_timestep l_timestep(*this, a_x[i], l_cy, l_hy);
                    l_cy = l_timestep.m_cy;
                    l_hy = l_timestep.m_y;
                    l_result[i] = l_timestep.m_y;
                }

                return l_result;

            }

            state_gradient_pair* parameterized_interpolate(
                const sgp_ptr_vector& a_x
            )
            {
                return multiply(
                    normalize(sigmoid(parameters(a_x.size()))),
                    a_x
                );
            }

            state_gradient_pair* vector_magnitude(
                const sgp_ptr_vector& a_x
            )
            {
                sgp_ptr_vector l_square_ys(a_x.size());
                for (int i = 0; i < a_x.size(); i++)
                {
                    l_square_ys[i] = pow(a_x[i], constant(2));
                }
                return pow(additive_aggregate(l_square_ys), constant(0.5));
            }

            state_gradient_pair* cosine_similarity(
                const sgp_ptr_vector& a_x_0,
                const sgp_ptr_vector& a_x_1
            )
            {
                auto l_multiply = multiply(a_x_0, a_x_1);
                auto l_magnitude_0 = vector_magnitude(a_x_0);
                auto l_magnitude_1 = vector_magnitude(a_x_1);
                return divide(divide(l_multiply, l_magnitude_0), l_magnitude_1);
            }

            state_gradient_pair* euclidian_distance(
                const sgp_ptr_vector& a_x_0,
                const sgp_ptr_vector& a_x_1
            )
            {
                return vector_magnitude(subtract(a_x_0, a_x_1));
            }

            sgp_ptr_vector similarity_interpolate(
                const sgp_ptr_vector& a_query,
                const sgp_ptr_matrix& a_keys,
                const sgp_ptr_matrix& a_values
            )
            {
                sgp_ptr_vector l_similarity_ys(a_keys.size()); // Each element between 0 and +inf

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

            sgp_ptr_vector convolve(
                const sgp_ptr_matrix& a_x,
                const sgp_ptr_matrix& a_filter,
                const size_t& a_stride = 1
            )
            {
                // Since the first dimension is considered to be height of the filter, we reserve the first dimension as
                // being non-spacial, and hence we use only the [0].size() as the width of our matrix.

                int l_right_most_position = a_x[0].size() - a_filter[0].size();

                assert(l_right_most_position >= 0);

                size_t l_convolution_count = (l_right_most_position / a_stride) + 1;

                sgp_ptr_vector l_result(l_convolution_count);

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

            sgp_ptr_matrix convolve(
                const sgp_ptr_cuboid& a_x,
                const sgp_ptr_cuboid& a_filter,
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

                sgp_ptr_matrix l_result(l_vertical_convolution_count);

                for (int i = 0; i < l_vertical_convolution_count; i++)
                {
                    sgp_ptr_vector l_result_row(l_horizontal_convolution_count);
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

            sgp_ptr_vector average_pool(
                const sgp_ptr_vector& a_x,
                const size_t& a_bin_width,
                const size_t& a_stride = 1
            )
            {
                size_t l_right_most_index = a_x.size() - a_bin_width;

                size_t l_pool_size = (l_right_most_index / a_stride) + 1;

                sgp_ptr_vector l_result(l_pool_size);

                for (int i = 0; i < l_pool_size; i++)
                {
                    l_result[i] = average(range(a_x, i * a_stride, a_bin_width));
                }

                return l_result;

            }

            sgp_ptr_matrix average_pool(
                const sgp_ptr_matrix& a_x,
                const size_t& a_bin_height,
                const size_t& a_bin_width,
                const size_t& a_stride = 1
            )
            {
                size_t l_top_most_index = a_x.size() - a_bin_height;
                size_t l_right_most_index = a_x[0].size() - a_bin_width;

                size_t l_pool_height = (l_top_most_index / a_stride) + 1;
                size_t l_pool_width = (l_right_most_index / a_stride) + 1;

                sgp_ptr_matrix l_result(l_pool_height);

                for (int i = 0; i < l_pool_height; i++)
                {
                    sgp_ptr_vector l_result_row(l_pool_width);

                    for (int j = 0; j < l_pool_width; j++)
                    {
                        l_result_row[j] = average(flatten(range(a_x, i * a_stride, j * a_stride, a_bin_height, a_bin_width)));
                    }

                    l_result[i] = l_result_row;

                }

                return l_result;

            }

            state_gradient_pair* mean_squared_error(
                state_gradient_pair* a_prediction,
                state_gradient_pair* a_desired
            )
            {
                auto l_error = subtract(a_prediction, a_desired);
                return pow(l_error, constant(2));
            }

            state_gradient_pair* mean_squared_error(
                const sgp_ptr_vector& a_prediction,
                const sgp_ptr_vector& a_desired
            )
            {
                assert(a_prediction.size() == a_desired.size());

                sgp_ptr_vector l_squared_errors(a_prediction.size());

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

            state_gradient_pair* mean_squared_error(
                const sgp_ptr_matrix& a_prediction,
                const sgp_ptr_matrix& a_desired
            )
            {
                assert(a_prediction.size() == a_desired.size());
                assert(a_prediction[0].size() == a_desired[0].size());

                sgp_ptr_matrix l_squared_errors(a_prediction.size());

                for (int i = 0; i < a_prediction.size(); i++)
                {
                    sgp_ptr_vector l_squared_error_row(a_prediction[0].size());
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

            state_gradient_pair* mean_squared_error(
                const sgp_ptr_cuboid& a_prediction,
                const sgp_ptr_cuboid& a_desired
            )
            {
                assert(a_prediction.size() == a_desired.size());
                assert(a_prediction[0].size() == a_desired[0].size());
                assert(a_prediction[0][0].size() == a_desired[0][0].size());

                sgp_ptr_vector l_squared_errors(a_prediction.size() * a_prediction[0].size() * a_prediction[0][0].size());

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

            state_gradient_pair* cross_entropy(
                state_gradient_pair* a_prediction,
                state_gradient_pair* a_desired
            )
            {
                auto l_first_term = multiply(a_desired, this->log(a_prediction));
                auto l_second_term = multiply(subtract(constant(1), a_desired), this->log(subtract(constant(1), a_prediction)));
                auto l_negated_sum = multiply(constant(-1), add(l_first_term, l_second_term));
                return l_negated_sum;
            }
            
        };


		class optimizer
		{
		private:
			bool m_normalize_gradients = false;

		public:
			sgp_ptr_vector m_values;

		public:
			optimizer(
				const sgp_ptr_vector& a_values,
				const bool& a_normalize_gradients
			) :
				m_values(a_values.begin(), a_values.end()),
				m_normalize_gradients(a_normalize_gradients)
			{

			}

			virtual void update(

			)
			{

			}

		protected:
			state_vector useful_gradients(

			)
			{
				state_vector l_gradients = get_gradient(m_values);

				if (m_normalize_gradients)
				{
					double l_normalization_denominator = 0;

					for (const auto& l_gradient : l_gradients)
						l_normalization_denominator += std::abs(l_gradient);

					for (auto& l_gradient : l_gradients)
						l_gradient /= l_normalization_denominator;

				}

				return l_gradients;

			}

		};

		class gradient_descent : public optimizer
		{
		public:
			double m_learn_rate = 0;

		public:
			gradient_descent(
				const sgp_ptr_vector& a_values,
				const bool& a_normalize_gradients,
				const double& a_learn_rate
			) :
				optimizer(a_values, a_normalize_gradients),
				m_learn_rate(a_learn_rate)
			{

			}

			virtual void update(

			)
			{
				state_vector l_gradients = useful_gradients();
				for (int i = 0; i < m_values.size(); i++)
				{
					m_values[i]->m_state -= m_learn_rate * l_gradients[i];
				}
			}

		};

		class gradient_descent_with_momentum : public optimizer
		{
		public:
			double m_learn_rate = 0;
			double m_beta = 0;
			double m_alpha = 0;
			state_vector m_momenta;

		public:
			gradient_descent_with_momentum(
				const sgp_ptr_vector& a_values,
				const bool& a_normalize_gradients,
				const double& a_learn_rate,
				const double& a_beta
			) :
				optimizer(a_values, a_normalize_gradients),
				m_learn_rate(a_learn_rate),
				m_beta(a_beta),
				m_alpha(1.0 - a_beta),
				m_momenta(a_values.size())
			{
				assert(a_beta >= 0 && a_beta <= 1);
			}

			virtual void update(

			)
			{
				state_vector l_gradients = useful_gradients();
				for (int i = 0; i < m_values.size(); i++)
				{
					auto& l_value = m_values[i];
					auto& l_momentum = m_momenta[i];
					l_momentum = m_beta * l_momentum + m_alpha * l_gradients[i];
					l_value->m_state -= m_learn_rate * l_momentum;
				}
			}

		};

	}
}

#endif
