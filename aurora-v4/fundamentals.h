#ifndef FUNDAMENTALS_H
#define FUNDAMENTALS_H

#include <vector>
#include <random>
#include <string>
#include <ostream>
#include <initializer_list>
#include <array>
#include <functional>

namespace aurora
{
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

    /// Defining some typedefs for improving the readability of code
    /// and user-friendliness.

    // We want a multidimensional array, the dimensions of which are defined at compile time.
    
    template<typename T, size_t I, size_t ... J>
    struct tensor : public std::array<tensor<T, J ...>, I>
    {
        
    };

    template<typename T, size_t I>
    struct tensor<T, I> : public std::array<T, I>
    {
        
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

    template<typename T, size_t I, size_t J>
    inline tensor<T, I> col(
        const tensor<T, I, J>& a_x,
        const size_t& a_col
    )
    {
        tensor<T, I> l_result;

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

    template<typename T, size_t I, size_t ... J>
    T mean_squared_error(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        return average(pow(flatten(subtract(a_x_0, a_x_1)), constant<T>(2.0)));
    }

    inline std::default_random_engine i_default_random_engine;

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
