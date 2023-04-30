#ifndef LATENT_H
#define LATENT_H

#include <vector>
#include <memory>
#include <assert.h>
#include <stdexcept>
#include <functional>
#include <ostream>
#include <istream>
#include "fundamentals.h"

namespace aurora
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
            return state_gradient_pair_dependency{m_state, *m_partial_gradients.back()};
        }
        
    };

    std::ostream& operator <<(std::ostream& a_out, const state_gradient_pair* a_state_gradient_pair)
    {
        a_out << a_state_gradient_pair->m_state;
        return a_out;
    }

    std::istream& operator >>(std::istream& a_in, state_gradient_pair* a_state_gradient_pair)
    {
        a_in >> a_state_gradient_pair->m_state;
        return a_in;
    }

    template<size_t I, size_t ... J>
    using latent_tensor = tensor<state_gradient_pair*, I, J ...>;

    template<size_t I, size_t ... J>
    tensor<state_gradient_pair*, I, J ...> input(
        const std::function<double()>& a_get_value
    )
    {
        return constant<state_gradient_pair*, I, J ...>(a_get_value);
    }

    template<size_t I, size_t ... J>
    tensor<state_gradient_pair*, I, J ...> input(
        const double& a_value = 0
    )
    {
        return constant<state_gradient_pair*, I, J ...>(a_value);
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
        static std::vector<model> s_models;
        std::vector<std::shared_ptr<element>> m_elements;

        model(

        )
        {
            
        }

    public:
        static void begin(

        )
        {
            s_models.push_back(model());
        }

        static model end(

        )
        {
            model l_result = s_models.back();
            s_models.pop_back();
            return l_result;
        }

        static void push(
            const std::shared_ptr<element>& a_element
        )
        {
            s_models.back().m_elements.push_back(a_element);
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

        const std::vector<std::shared_ptr<element>>& elements(

        ) const
        {
            return m_elements;
        }

    };

    std::vector<model> model::s_models;

    template<>
    inline state_gradient_pair* constant<state_gradient_pair*>(
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;

    }

    template<>
    inline state_gradient_pair* add<state_gradient_pair*>(
        state_gradient_pair* const& a_x_0,
        state_gradient_pair* const& a_x_1
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;
        
    }

    template<>
    inline state_gradient_pair* subtract<state_gradient_pair*>(
        state_gradient_pair* const& a_x_0,
        state_gradient_pair* const& a_x_1
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;
        
    }

    template<>
    inline state_gradient_pair* multiply<state_gradient_pair*>(
        state_gradient_pair* const& a_x_0,
        state_gradient_pair* const& a_x_1
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;

    }

    template<>
    inline state_gradient_pair* divide<state_gradient_pair*>(
        state_gradient_pair* const& a_x_0,
        state_gradient_pair* const& a_x_1
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;
        
    }

    template<>
    inline state_gradient_pair* pow<state_gradient_pair*>(
        state_gradient_pair* const& a_x_0,
        state_gradient_pair* const& a_x_1
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;

    }
    
    template<>
    inline state_gradient_pair* sigmoid<state_gradient_pair*>(
        state_gradient_pair* const& a_x
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;
        
    }

    template<>
    inline state_gradient_pair* tanh<state_gradient_pair*>(
        state_gradient_pair* const& a_x
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;
        
    }

    template<>
    inline state_gradient_pair* leaky_relu<state_gradient_pair*>(
        state_gradient_pair* const& a_x,
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;

    }

    template<>
    inline state_gradient_pair* log<state_gradient_pair*>(
        state_gradient_pair* const& a_x
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
        
        model::push(std::shared_ptr<element>(l_element));

        return &l_element->m_y;
        
    }

    template<size_t I>
    class optimizer
    {
    private:
    	bool m_normalize_gradients = false;

    public:
    	latent_tensor<I> m_values;

    public:
    	optimizer(
    		const latent_tensor<I>& a_values,
    		const bool& a_normalize_gradients
    	) :
    		m_values(a_values),
    		m_normalize_gradients(a_normalize_gradients)
    	{

    	}

    	virtual void update(

    	)
    	{

    	}

    protected:
    	tensor<double, I> useful_gradients(

    	)
    	{
    		tensor<double, I> l_gradients = get_gradient(m_values);

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

    template<size_t I>
    class gradient_descent : public optimizer<I>
    {
    public:
    	double m_learn_rate = 0;

    public:
    	gradient_descent(
    		const latent_tensor<I>& a_values,
    		const bool& a_normalize_gradients,
    		const double& a_learn_rate
    	) :
    		optimizer<I>(a_values, a_normalize_gradients),
    		m_learn_rate(a_learn_rate)
    	{

    	}

    	virtual void update(

    	)
    	{
    		tensor<double, I> l_gradients = this->useful_gradients();
    		for (int i = 0; i < this->m_values.size(); i++)
    		{
    			this->m_values[i]->m_state -= this->m_learn_rate * l_gradients[i];
    		}
    	}

    };

    template<size_t I>
    class gradient_descent_with_momentum : public gradient_descent<I>
    {
    public:
    	double m_beta = 0;
    	double m_alpha = 0;
    	tensor<double, I> m_momenta;

    public:
    	gradient_descent_with_momentum(
    		const latent_tensor<I>& a_values,
    		const bool& a_normalize_gradients,
    		const double& a_learn_rate,
    		const double& a_beta
    	) :
    		gradient_descent<I>(a_values, a_normalize_gradients, a_learn_rate),
    		m_beta(a_beta),
    		m_alpha(1.0 - a_beta),
            m_momenta(constant<double, I>())
    	{
    		assert(a_beta >= 0 && a_beta <= 1);
    	}

    	virtual void update(

    	)
    	{
    		tensor<double, I> l_gradients = this->useful_gradients();
    		for (int i = 0; i < this->m_values.size(); i++)
    		{
    			auto& l_value = this->m_values[i];
    			auto& l_momentum = this->m_momenta[i];
    			l_momentum = m_beta * l_momentum + m_alpha * l_gradients[i];
    			l_value->m_state -= this->m_learn_rate * l_momentum;
    		}
    	}

    };

    inline double get_state(
        state_gradient_pair* a_sgp_ptr
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

}

#endif
