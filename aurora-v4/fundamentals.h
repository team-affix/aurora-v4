#ifndef FUNDAMENTALS_H
#define FUNDAMENTALS_H

#include <vector>
#include <random>
#include <string>
#include <ostream>
#include <initializer_list>
#include <array>
#include <functional>
#include <assert.h>

namespace aurora
{
    inline std::mt19937 i_random_engine;

    template<typename T>
    inline T constant(
        const double& a_x = 0
    );

    template<typename T>
    inline T add(
        const T& a_x_0,
        const T& a_x_1
    );

    template<typename T>
    inline T subtract(
        const T& a_x_0,
        const T& a_x_1
    );

    template<typename T>
    inline T multiply(
        const T& a_x_0,
        const T& a_x_1
    );

    template<typename T>
    inline T divide(
        const T& a_x_0,
        const T& a_x_1
    );

    template<typename T>
    inline T sigmoid(
        const T& a_x
    );

    template<typename T>
    inline T tanh(
        const T& a_x
    );

    template<typename T>
    inline T leaky_relu(
        const T& a_x,
        const double& a_m
    );

    template<typename T>
    inline T log(
        const T& a_x
    );

    template<typename T>
    inline T pow(
        const T& a_x_0,
        const T& a_x_1
    );

    template<typename T>
    inline T abs(
        const T& a_x
    );

    template<>
    inline double constant<double>(
        const double& a_x
    )
    {
        return a_x;
    }

    template<>
    inline double add<double>(
        const double& a_x_0,
        const double& a_x_1
    )
    {
        return a_x_0 + a_x_1;
    }

    template<>
    inline double subtract<double>(
        const double& a_x_0,
        const double& a_x_1
    )
    {
        return a_x_0 - a_x_1;
    }

    template<>
    inline double multiply<double>(
        const double& a_x_0,
        const double& a_x_1
    )
    {
        return a_x_0 * a_x_1;
    }

    template<>
    inline double divide<double>(
        const double& a_x_0,
        const double& a_x_1
    )
    {
        return a_x_0 / a_x_1;
    }

    template<>
    inline double sigmoid<double>(
        const double& a_x
    )
    {
        return 1.0 / (1.0 + std::exp(-a_x));
    }

    template<>
    inline double tanh<double>(
        const double& a_x
    )
    {
        return std::tanh(a_x);
    }
    
    template<>
    inline double leaky_relu<double>(
    	const double& a_x,
    	const double& a_m
    )
    {
    	if (a_x >= 0)
    		return a_x;
    	else
    		return a_m * a_x;
    }

    template<>
    inline double log<double>(
        const double& a_x
    )
    {
        return std::log(a_x);
    }

    template<>
    inline double pow<double>(
        const double& a_x_0,
        const double& a_x_1
    )
    {
        return std::pow(a_x_0, a_x_1);
    }

    template<>
    inline double abs<double>(
        const double& a_x
    )
    {
        return std::abs(a_x);
    }

    /// Defining some typedefs for improving the readability of code
    /// and user-friendliness.

    // We want a multidimensional array, the dimensions of which are defined at compile time.

    template<typename T, size_t I, size_t ... J>
    struct tensor : public std::array<tensor<T, J ...>, I>
    {
        constexpr size_t flattened_size(

        )
        {
            return (I * ... * J);
        }
        
        T* flattened_begin(

        )
        {
            return (T*)this;
        }

        const T* flattened_begin(

        ) const
        {
            return (const T*)this;
        }

        T* flattened_end(

        )
        {
            return flattened_begin() + flattened_size();
        }

        const T* flattened_end(

        ) const
        {
            return flattened_begin() + flattened_size();
        }

    };

    template<typename T, size_t I>
    struct tensor<T, I> : public std::array<T, I>
    {
        constexpr size_t flattened_size(

        )
        {
            return I;
        }

        T* flattened_begin(

        )
        {
            return (T*)this;
        }
        
        const T* flattened_begin(

        ) const
        {
            return (T*)this;
        }

        T* flattened_end(

        )
        {
            return flattened_begin() + flattened_size();
        }

        const T* flattened_end(

        ) const
        {
            return flattened_begin() + flattened_size();
        }
        
    };

    template<typename T>
    inline T constant(
        const std::function<double()>& a_get_value
    )
    {
        // Since this is the zeroth-order call, just call a_get_value.
        return constant<T>(a_get_value());
    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> constant(
        const std::function<double()>& a_get_value
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = constant<T, J ...>(a_get_value);
        
        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> constant(
        const double& a_double = 0
    )
    {
        return constant<T, I, J ...>([a_double]{return a_double;});
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
    inline tensor<T, B, I/B, J ...> partition(
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
    inline tensor<T, I1+I2, J ...> concat(
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

    template<typename T, size_t I1, size_t ... Is, size_t ... J>
        requires (sizeof...(Is) > 1)
    inline tensor<T, (I1 + ... + Is), J ...> concat(
        const tensor<T, I1, J ...>& a_x_0,
        const tensor<T, Is, J ...>& ... a_xs
    )
    {
        return concat(a_x_0, concat(a_xs...));
    }

    template<typename T, size_t I>
    inline tensor<T, I> flatten(
        const tensor<T, I>& a_tensor
    )
    {
        return a_tensor;
    }

    template<typename T, size_t I, size_t J>
    inline tensor<T, I * J> flatten(
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
        requires(sizeof...(J) > 1)
    inline tensor<T, (I * ... * J)> flatten(
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

    inline auto flatten(
        const auto& a_tensor_0,
        const auto& ... a_tensors
    )
    {
        return concat(flatten(a_tensor_0), flatten(a_tensors...));
    }

    template<size_t SRC_BEGIN_INDEX = 0, typename T, size_t I1, size_t ... J1, size_t I2, size_t ... J2>
    inline void copy(
        const tensor<T, I1, J1 ...>& a_source,
        tensor<T, I2, J2 ...>& a_destination_0,
        auto& ... a_destinations
    )
    {
        constexpr size_t SOURCE_ELEMENT_COUNT = (I1 * ... * J1);
        constexpr size_t FIRST_DESTINATION_ELEMENT_COUNT = (I2 * ... * J2);
        
        const T* l_src_begin = a_source.flattened_begin() + SRC_BEGIN_INDEX;
        const T* l_src_end   = l_src_begin + FIRST_DESTINATION_ELEMENT_COUNT;
        T* l_dst = a_destination_0.flattened_begin();

        // This takes advantage of the fact that std::array stores elements
        // in contiguous memory.
        std::copy(l_src_begin, l_src_end, l_dst);
        
        if constexpr (sizeof...(a_destinations) > 0)
        {
            copy<SRC_BEGIN_INDEX + FIRST_DESTINATION_ELEMENT_COUNT>(a_source, a_destinations ...);
        }
        else
        {
            // This copy was the last.
            static_assert(SRC_BEGIN_INDEX + FIRST_DESTINATION_ELEMENT_COUNT == SOURCE_ELEMENT_COUNT, "Inequal number of elements in copy.");
        }

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> add(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = add(a_x_0[i], a_x_1[i]);

        return l_result;

    }

    template<typename T, size_t I>
        requires (I > 0)
    inline T additive_aggregate(
        const tensor<T, I>& a_tensor
    )
    {
        T l_result = a_tensor[0];

        for (int i = 1; i < I; i++)
            l_result = add(l_result, a_tensor[i]);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
        requires (I > 0)
    inline tensor<T, J ...> additive_aggregate(
        const tensor<T, I, J ...>& a_tensor
    )
    {
        tensor<T, J ...> l_result = a_tensor[0];

        for (int i = 1; i < I; i++)
            l_result = add(l_result, a_tensor[i]);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> subtract(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = subtract(a_x_0[i], a_x_1[i]);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> abs(
        const tensor<T, I, J ...>& a_x
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = abs(a_x[i]);

        return l_result;
        
    }

    /// @brief Tensor-tensor multiplication.
    /// @tparam T 
    /// @tparam I 
    /// @tparam ...J 
    /// @param a_x_0 
    /// @param a_x_1 
    /// @return 
    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> multiply(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = multiply(a_x_0[i], a_x_1[i]);

        return l_result;
        
    }

    /// @brief Tensor-scalar multiplication.
    /// @tparam T 
    /// @tparam I 
    /// @tparam ...J 
    /// @param a_x_0 
    /// @param a_x_1 
    /// @return 
    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> multiply(
        const tensor<T, I, J ...>& a_x_0,
        const T& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = multiply(a_x_0[i], a_x_1);

        return l_result;

    }

    template<typename T, size_t I, size_t J>
    inline tensor<T, I> multiply(
        const tensor<T, I, J>& a_x_0,
        const tensor<T, J>& a_x_1
    )
    {
        return flatten(dot(a_x_0, partition<J>(a_x_1)));
    }

    template<typename T, size_t I>
    inline T average(
        const tensor<T, I>& a_x
    )
    {
        return divide(additive_aggregate(a_x), constant<T>(I));
    }

    template<typename T, size_t I>
    inline T stddev(
        const tensor<T, I>& a_x
    )
    {
        T l_avg = average(a_x);
        auto l_devs = subtract(a_x, constant<T, I>(l_avg));
        auto l_abs_devs = abs(l_devs);
        return average(l_abs_devs);
    }

    template<typename T, size_t I, size_t ... J>
        requires (sizeof...(J) > 0)
    inline tensor<T, J ...> average(
        const tensor<T, I, J ...>& a_x
    )
    {
        return multiply(additive_aggregate(a_x), constant<T>(1.0 / double(I)));
    }

    template<typename T, size_t I, size_t J>
    inline tensor<T, J> row(
        const tensor<T, I, J>& a_x,
        const size_t& a_row
    )
    {
        return a_x[a_row];
    }

    template<typename T, size_t I, size_t J, size_t ... K>
    inline tensor<T, I, K ...> col(
        const tensor<T, I, J, K ...>& a_x,
        const size_t& a_col
    )
    {
        tensor<T, I, K ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = a_x[i][a_col];

        return l_result;

    }

    template<typename T, size_t I>
    inline T dot(
        const tensor<T, I>& a_x_0,
        const tensor<T, I>& a_x_1
    )
    {
        T l_result = constant<T>(0.0);

        for (int i = 0; i < I; i++)
            l_result = add(l_result, multiply(a_x_0[i], a_x_1[i]));

        return l_result;

    }

    template<typename T, size_t I1, size_t J1, size_t J2>
    inline tensor<T, I1, J2> dot(
        const tensor<T, I1, J1>& a_x_0,
        const tensor<T, J1, J2>& a_x_1
    )
    {
        tensor<T, I1, J2> l_result;

        for (int i = 0; i < I1; i++)
            for (int j = 0; j < J2; j++)
                l_result[i][j] = dot(row(a_x_0, i), col(a_x_1, j));

        return l_result;

    }

    /// @brief This function currently flips the tensor's outermost two ranks.
    /// @tparam I 
    /// @tparam J 
    /// @tparam K 
    /// @tparam ...L 
    /// @param a_tensor 
    /// @return 
    template<typename T, size_t I, size_t J, size_t ... K>
    inline tensor<T, J, I, K ...> transpose(
        const tensor<T, I, J, K ...>& a_tensor
    )
    {
        tensor<T, J, I, K ...> l_result;

        for (int i = 0; i < I; i++)
            for (int j = 0; j < J; j++)
                l_result[j][i] = a_tensor[i][j];

        return l_result;

    }

    template<typename T>
    inline T negate(
        const T& a_x
    )
    {
        return multiply(a_x, constant<T>(-1.0));
    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> negate(
        const tensor<T, I, J ...>& a_x
    )
    {
        return multiply(a_x, constant<T>(-1.0));
    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> sigmoid(
        const tensor<T, I, J ...>& a_tensor
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = sigmoid(a_tensor[i]);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> tanh(
        const tensor<T, I, J ...>& a_tensor
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = tanh(a_tensor[i]);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> leaky_relu(
        const tensor<T, I, J ...>& a_tensor,
        const double& a_m
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = leaky_relu(a_tensor[i], a_m);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> log(
        const tensor<T, I, J ...>& a_tensor
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = log(a_tensor[i]);

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
    inline tensor<T, I, J ...> pow(
        const tensor<T, I, J ...>& a_x_0,
        const T& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = pow(a_x_0[i], a_x_1);

        return l_result;

    }

    template<size_t J2, typename T, size_t J1>
    inline void lstm_timestep(
        const tensor<T, J1>& a_x,
        const tensor<T, J2>& a_cx,
        const tensor<T, J2>& a_hx,
        const tensor<T, J2>& a_forget_gate_bias,
        const tensor<T, J2>& a_input_limit_gate_bias,
        const tensor<T, J2>& a_input_gate_bias,
        const tensor<T, J2>& a_output_gate_bias,
        const tensor<T, J2, J1+J2>& a_forget_gate_weights,
        const tensor<T, J2, J1+J2>& a_input_limit_gate_weights,
        const tensor<T, J2, J1+J2>& a_input_gate_weights,
        const tensor<T, J2, J1+J2>& a_output_gate_weights,
        tensor<T, J2>& a_cy,
        tensor<T, J2>& a_hy
    )
    {
        auto l_hx_x_concat = concat(a_hx, a_x);

        // Construct gates

        auto l_forget_gate = sigmoid(add(multiply(a_forget_gate_weights, l_hx_x_concat), a_forget_gate_bias));

        auto l_input_limit_gate = sigmoid(add(multiply(a_input_limit_gate_weights, l_hx_x_concat), a_input_limit_gate_bias));

        auto l_input_gate = tanh(add(multiply(a_input_gate_weights, l_hx_x_concat), a_input_gate_bias));

        auto l_output_gate = sigmoid(add(multiply(a_output_gate_weights, l_hx_x_concat), a_output_gate_bias));

        // Forget parts of the cell state
        tensor<T, J2> l_cell_state_after_forget = multiply(a_cx, l_forget_gate);

        // Calculate the input to the cell state
        tensor<T, J2> l_limited_input = multiply(l_input_gate, l_input_limit_gate);

        // Write the input to the cell state
        tensor<T, J2> l_cell_state_after_input = add(l_cell_state_after_forget, l_limited_input);

        // Cell state is now finalized, save it as the cell state output
        a_cy = l_cell_state_after_input;

        // Do a temporary step to compute tanh(cy)
        tensor<T, J2> l_cell_state_after_tanh = tanh(l_cell_state_after_input);

        // Compute output to the timestep
        a_hy = multiply(l_output_gate, l_cell_state_after_tanh);

    }

    template<size_t J2, typename T, size_t I, size_t J1>
    inline tensor<T, I, J2> lstm(
        const tensor<T, I, J1>& a_x,
        const tensor<T, J2>& a_cx,
        const tensor<T, J2>& a_hx,
        const tensor<T, J2>& a_forget_gate_bias,
        const tensor<T, J2>& a_input_limit_gate_bias,
        const tensor<T, J2>& a_input_gate_bias,
        const tensor<T, J2>& a_output_gate_bias,
        const tensor<T, J2, J1+J2>& a_forget_gate_weights,
        const tensor<T, J2, J1+J2>& a_input_limit_gate_weights,
        const tensor<T, J2, J1+J2>& a_input_gate_weights,
        const tensor<T, J2, J1+J2>& a_output_gate_weights
    )
    {
        tensor<T, I, J2> l_result;

        tensor<T, J2> l_cy = a_cx;
        tensor<T, J2> l_hy = a_hx;

        for (int i = 0; i < a_x.size(); i++)
        {
            lstm_timestep(
                a_x[i],
                l_cy,
                l_hy,
                a_forget_gate_bias,
                a_input_limit_gate_bias,
                a_input_gate_bias,
                a_output_gate_bias,
                a_forget_gate_weights,
                a_input_limit_gate_weights,
                a_input_gate_weights,
                a_output_gate_weights,
                l_cy,
                l_hy
            );
            l_result[i] = l_hy;
        }

        return l_result;

    }

    // state_gradient_pair* cross_entropy(
    //     state_gradient_pair* a_prediction,
    //     state_gradient_pair* a_desired
    // )
    // {
    //     auto l_first_term = multiply(a_desired, this->log(a_prediction));
    //     auto l_second_term = multiply(subtract(constant(1), a_desired), this->log(subtract(constant(1), a_prediction)));
    //     auto l_negated_sum = multiply(constant(-1), add(l_first_term, l_second_term));
    //     return l_negated_sum;
    // }

    template<typename T, size_t I>
    inline T magnitude(
        const tensor<T, I>& a_tensor
    )
    {
        T l_sum = constant<T>(0);

        for (const T& l_element : a_tensor)
        {
            l_sum = add(l_sum, pow(l_element, constant<T>(2.0)));
        }

        return sqrt(l_sum);

    }

    template<typename T, size_t I>
    inline tensor<T, I> normalize(
        const tensor<T, I>& a_tensor
    )
    {
        return multiply(a_tensor, divide(constant<T>(1.0), magnitude(a_tensor)));
    }

    template<typename T, size_t I>
    inline T cos_sim(
        const tensor<T, I>& a_x_0,
        const tensor<T, I>& a_x_1
    )
    {
        return divide(dot(a_x_0, a_x_1), multiply(magnitude(a_x_0), magnitude(a_x_1)));
    }

    template<typename T, size_t I>
    inline T euclidian_distance(
        const tensor<T, I>& a_x_0,
        const tensor<T, I>& a_x_1
    )
    {
        return magnitude(subtract(a_x_0, a_x_1));
    }

    template<typename T>
    inline T squared_error(
        const T& a_x_0,
        const T& a_x_1
    )
    {
        return pow(subtract(a_x_0, a_x_1), constant<T>(2.0));
    }

    template<typename T, size_t I, size_t ... J>
    inline T mean_squared_error(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        return average(pow(flatten(subtract(a_x_0, a_x_1)), constant<T>(2.0)));
    }

    template<typename T>
    inline T log_loss(
        const T& a_label,
        const T& a_prediction
    )
    {
        T l_label_compliment = add(constant<T>(1.0), negate(a_label));
        T l_prediction_compliment = add(constant<T>(1.0), negate(a_prediction));
        
        return negate(add(multiply(a_label, log(a_prediction)), multiply(l_label_compliment, log(l_prediction_compliment))));

    }

    template<size_t PARTICLE_COUNT, size_t I>
    class particle_swarm_optimizer
    {
    private:
        static std::uniform_real_distribution<double> s_urd;
    
    private:
        tensor<double, PARTICLE_COUNT, I>& m_positions;
        tensor<double, PARTICLE_COUNT, I>  m_local_best_positions;
        tensor<double, PARTICLE_COUNT, I>  m_velocities;
        tensor<double, PARTICLE_COUNT>     m_local_best_rewards;

        double m_w;
        double m_c1;
        double m_c2;
        double m_memory;
        double m_global_best_reward;
        tensor<double, I> m_global_best_position;

    public:
        particle_swarm_optimizer(
            tensor<double, PARTICLE_COUNT, I>& a_positions,
            const double& a_w,
            const double& a_c1,
            const double& a_c2,
            const double& a_memory = 1.0
        ) :
            m_positions(a_positions),
            m_local_best_positions(constant<double, PARTICLE_COUNT, I>()),
            m_velocities(constant<double, PARTICLE_COUNT, I>()),
            m_local_best_rewards(constant<double, PARTICLE_COUNT>()),
            m_w(a_w),
            m_c1(a_c1),
            m_c2(a_c2),
            m_memory(a_memory),
            m_global_best_reward(-INFINITY),
            m_global_best_position(constant<double, I>())
        {

        }

        void update(
            const tensor<double, PARTICLE_COUNT>& a_particle_rewards
        )
        {
            // This adds a memory decay to the global best reward, to
            // assist with non-static or random environments.
            m_global_best_reward *= m_memory;
            
            // Get the global best position if it has improved
            for (int i = 0; i < PARTICLE_COUNT; i++)
            {
                if (a_particle_rewards[i] > m_global_best_reward)
                {
                    m_global_best_reward = a_particle_rewards[i];
                    m_global_best_position = m_positions[i];
                }
            }

            // Update all particle positions
            for (int i = 0; i < PARTICLE_COUNT; i++)
            {
                update(
                    m_positions[i],
                    m_local_best_positions[i],
                    m_velocities[i],
                    m_local_best_rewards[i],
                    a_particle_rewards[i]
                );
            }

        }

        double global_best_reward(

        )
        {
            return m_global_best_reward;
        }

        tensor<double, I> global_best_position(

        )
        {
            return m_global_best_position;
        }

    private:
        void update(
            tensor<double, I>& a_position,
            tensor<double, I>& a_local_best_position,
            tensor<double, I>& a_velocity,
            double& a_local_best_reward,
            const double& a_reward
        )
        {
            if (a_reward > a_local_best_reward)
            {
                a_local_best_position = a_position;
                a_local_best_reward = a_reward;
            }

            tensor<double, I> l_weighted_particle_velocity = multiply(a_velocity, m_w);
            tensor<double, I> l_cognitive_term = multiply(multiply(subtract(a_local_best_position, a_position), m_c1), s_urd(i_random_engine));
            tensor<double, I> l_social_term = multiply(multiply(subtract(m_global_best_position, a_position), m_c2), s_urd(i_random_engine));
            a_velocity = add(add(l_weighted_particle_velocity, l_cognitive_term), l_social_term);
            a_position = add(a_position, a_velocity);

        }

    };

    template<size_t PARTICLE_COUNT, size_t I>
    std::uniform_real_distribution<double> particle_swarm_optimizer<PARTICLE_COUNT, I>::s_urd(0, 1);

    template<size_t PARTICLE_COUNT, size_t I>
    class icpso
    {
    private:
        static std::uniform_real_distribution<double> s_urd;
    
    private:
        tensor<std::vector<double>, PARTICLE_COUNT, I> m_positions;
        tensor<size_t, PARTICLE_COUNT, I>              m_candidate_solutions;
        tensor<std::vector<double>, PARTICLE_COUNT, I> m_local_best_positions;
        tensor<std::vector<double>, PARTICLE_COUNT, I> m_velocities;
        tensor<double, PARTICLE_COUNT>                 m_local_best_rewards;

        double                         m_w;
        double                         m_c1;
        double                         m_c2;
        double                         m_epsilon;
        double                         m_epsilon_compliment;
        double                         m_global_best_reward;
        tensor<std::vector<double>, I> m_global_best_position;
        tensor<size_t, I>              m_global_best_solution;

    public:
        icpso(
            const tensor<size_t, I>& a_distribution_sizes,
            const double& a_w,
            const double& a_c1,
            const double& a_c2,
            const double& a_epsilon
        ) :
            m_local_best_rewards(constant<double, PARTICLE_COUNT>()),
            m_w(a_w),
            m_c1(a_c1),
            m_c2(a_c2),
            m_epsilon(a_epsilon),
            m_epsilon_compliment(1.0 - a_epsilon),
            m_global_best_reward(-INFINITY)
        {
            assert(a_w > 0 && a_w < 1);
            assert(a_c1 > 0 && a_c1 < 1);
            assert(a_c2 > 0 && a_c2 < 1);
            assert(a_epsilon > 0 && a_epsilon < 1);
            
            for (int i = 0; i < PARTICLE_COUNT; i++)
                for (int j = 0; j < I; j++)
                {
                    // Create distributions. (Not initializing values)
                    m_positions[i][j] = std::vector<double>(a_distribution_sizes[j]);
                    m_local_best_positions[i][j] = std::vector<double>(a_distribution_sizes[j]);
                    m_velocities[i][j] = std::vector<double>(a_distribution_sizes[j]);

                    // INITIALIZE THE POSITION DISTRIBUTIONS RANDOMLY
                    for (int k = 0; k < a_distribution_sizes[j]; k++)
                        m_positions[i][j][k] = s_urd(i_random_engine);

                    // Clip and normalize the distribution for this variable for this particle.
                    clip_and_normalize(m_positions[i][j]);

                }

            // Initialize the global best position.
            for (int i = 0; i < I; i++)
                m_global_best_position[i] = std::vector<double>(a_distribution_sizes[i]);

        }

        const tensor<size_t, PARTICLE_COUNT, I>& candidate_solutions(

        )
        {
            for (int i = 0; i < PARTICLE_COUNT; i++)
                for (int j = 0; j < I; j++)
                    m_candidate_solutions[i][j] = sample(m_positions[i][j]);

            return m_candidate_solutions;
            
        }

        void update(
            const tensor<double, PARTICLE_COUNT>& a_particle_rewards
        )
        {
            // Get the maximum immediate reward
            auto l_max_reward = std::max_element(a_particle_rewards.begin(), a_particle_rewards.end());
            size_t l_max_reward_index = l_max_reward - a_particle_rewards.begin();

            // Update global best position and reward if a new best exists
            if (*l_max_reward > m_global_best_reward)
            {
                update_best_position(
                    m_global_best_position,
                    m_positions[l_max_reward_index],
                    m_candidate_solutions[l_max_reward_index]
                );
                m_global_best_reward = *l_max_reward;
                m_global_best_solution = m_candidate_solutions[l_max_reward_index];
            }

            // Update all particle positions
            for (int i = 0; i < PARTICLE_COUNT; i++)
            {
                update(
                    m_positions[i],
                    m_candidate_solutions[i],
                    m_local_best_positions[i],
                    m_velocities[i],
                    m_local_best_rewards[i],
                    a_particle_rewards[i]
                );
            }

        }

        double global_best_reward(

        )
        {
            return m_global_best_reward;
        }

        const tensor<size_t, I>& global_best_solution(

        )
        {
            return m_global_best_solution;
        }

    private:
        void update_best_position(
            tensor<std::vector<double>, I>& a_old_best_position,
            const tensor<std::vector<double>, I>& a_new_best_position,
            const tensor<size_t, I>& a_candidate_solution
        )
        {
            for (int i = 0; i < I; i++)
                update_best_distribution(a_old_best_position[i], a_new_best_position[i], a_candidate_solution[i]);
        }

        void update_best_distribution(
            std::vector<double>& a_old_best_distribution,
            const std::vector<double>& a_new_best_distribution,
            const size_t& a_selected_variable_index
        )
        {
            ////////////////////////
            // FOR EACH PROBABILITY IN THE DISTRIBUTION WHICH
            // WAS NOT SAMPLED, DECREASE ITS VALUE (using decay so it will never be negative).
            // AND WHATEVER PROBABILITY VALUE IS LOST FOR EACH PROBABILITY,
            // INCREASE THE PROBABILITY OF CHOOSING THE SELECTED VARIABLE AGAIN.
            ////////////////////////

            a_old_best_distribution[a_selected_variable_index] = a_new_best_distribution[a_selected_variable_index];

            for (int i = 0; i < a_old_best_distribution.size(); i++)
            {
                if (i == a_selected_variable_index)
                    continue;
                
                a_old_best_distribution[i] = m_epsilon * a_new_best_distribution[i];
                a_old_best_distribution[a_selected_variable_index] += m_epsilon_compliment * a_new_best_distribution[i];
                
            }
            
        }
    
        void update(
            tensor<std::vector<double>, I>& a_position,
            const tensor<size_t, I>&        a_candidate_solution,
            tensor<std::vector<double>, I>& a_local_best_position,
            tensor<std::vector<double>, I>& a_velocity,
            double&                         a_local_best_reward,
            const double&                   a_reward
        )
        {
            if (a_reward > a_local_best_reward)
            {
                update_best_position(a_local_best_position, a_position, a_candidate_solution);
                a_local_best_reward = a_reward;
            }

            // Generate two random values
            double l_cognitive_coefficient = m_c1 * s_urd(i_random_engine);
            double l_social_coefficient = m_c2 * s_urd(i_random_engine);

            ////////////////////////////////////
            // UPDATE THE VELOCITY AND POSITION VECTORS
            ////////////////////////////////////
            for (int i = 0; i < PARTICLE_COUNT; i++)
                for (int j = 0; j < I; j++)
                    for (int k = 0; k < m_positions[i][j].size(); k++)
                    {
                        double& l_velocity = m_velocities[i][j][k];
                        double& l_position = m_positions[i][j][k];
                        const double& l_local_best_position = m_local_best_positions[i][j][k];
                        const double& l_global_best_position = m_global_best_position[j][k];

                        // UPDATE VELOCOTY VALUE
                        l_velocity = 
                            m_w * l_velocity + 
                            l_cognitive_coefficient * (l_local_best_position - l_position) +
                            l_social_coefficient * (l_global_best_position - l_position);

                        // UPDATE POSITION VALUE
                        l_position += l_velocity;
                        
                    }
            
        }

        static size_t sample(
            const std::vector<double>& a_distribution
        )
        {
            double l_remainder = s_urd(i_random_engine);

            size_t i = 0;

            for (; i < a_distribution.size() && l_remainder > 0; l_remainder -= a_distribution[i], i++);
            
            return i - 1;
            
        }

        static void clip_and_normalize(
            std::vector<double>& a_distribution
        )
        {
            double l_normalization_denominator = 0;

            for (int i = 0; i < a_distribution.size(); i++)
            {
                // Clip the value in the distribution.
                a_distribution[i] = std::min(std::max(a_distribution[i], 0.0), 1.0);

                l_normalization_denominator += a_distribution[i];
                
            }

            for (int i = 0; i < a_distribution.size(); i++)
                a_distribution[i] /= l_normalization_denominator;
            
        }

    };

    template<size_t PARTICLE_COUNT, size_t I>
    std::uniform_real_distribution<double> icpso<PARTICLE_COUNT, I>::s_urd(0, 1);

}

template<typename T, size_t I>
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element << ' ';

    return a_ostream;

}

template<typename T, size_t I, size_t ... J>
    requires (sizeof...(J) > 0)
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I, J ...>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element << '\n';

    return a_ostream;

}

template<typename T, size_t I, size_t ... J>
std::istream& operator>>(std::istream& a_istream, aurora::tensor<T, I, J ...>& a_tensor)
{
    for (auto& l_element : a_tensor)
        a_istream >> l_element;

    return a_istream;
    
}

#endif
