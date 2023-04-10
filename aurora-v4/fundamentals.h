#ifndef FUNDAMENTALS_H
#define FUNDAMENTALS_H

#include <vector>
#include <random>
#include <string>
#include <ostream>
#include <initializer_list>
#include <array>

namespace aurora
{
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
        tensor(

        )
        {
            std::fill_n(std::array<T, I>::begin(), std::array<T, I>::size(), T());
        }
    };

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

    template<typename T, size_t I>
    tensor<T, I> add(
        const tensor<T, I>& a_x_0,
        const tensor<T, I>& a_x_1
    )
    {
        tensor<T, I> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = a_x_0[i] + a_x_1[i];

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
        requires (sizeof...(J) > 0)
    tensor<T, I, J ...> add(
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
    T additive_aggregate(
        const tensor<T, I>& a_tensor
    )
    {
        T l_result = a_tensor[0];

        for (int i = 1; i < I; i++)
            l_result = l_result + a_tensor[i];

        return l_result;

    }

    template<typename T, size_t I, size_t ... J>
        requires (I > 0)
    tensor<T, J ...> additive_aggregate(
        const tensor<T, I, J ...>& a_tensor
    )
    {
        tensor<T, J ...> l_result = a_tensor[0];

        for (int i = 1; i < I; i++)
            l_result = add(l_result, a_tensor[i]);

        return l_result;

    }

    template<typename T, size_t I>
    tensor<T, I> subtract(
        const tensor<T, I>& a_x_0,
        const tensor<T, I>& a_x_1
    )
    {
        tensor<T, I> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = a_x_0[i] - a_x_1[i];

        return l_result;
            
    }

    template<typename T, size_t I, size_t ... J>
        requires (sizeof...(J) > 0)
    tensor<T, I, J ...> subtract(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = subtract(a_x_0[i], a_x_1[i]);

        return l_result;

    }

    /// @brief Vector-scalar multiplication.
    /// @tparam T 
    /// @tparam I 
    /// @param a_x_0 
    /// @param a_x_1 
    /// @return 
    template<typename T, size_t I>
    tensor<T, I> multiply(
        const tensor<T, I>& a_x_0,
        const T& a_x_1
    )
    {
        tensor<T, I> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = a_x_0[i] * a_x_1;

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
        requires (sizeof...(J) > 0)
    tensor<T, I, J ...> multiply(
        const tensor<T, I, J ...>& a_x_0,
        const T& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = multiply(a_x_0[i], a_x_1);

        return l_result;

    }

    /// @brief Vector-vector multiplication.
    /// @tparam T 
    /// @tparam I 
    /// @param a_x_0 
    /// @param a_x_1 
    /// @return 
    template<typename T, size_t I>
    tensor<T, I> multiply(
        const tensor<T, I>& a_x_0,
        const tensor<T, I>& a_x_1
    )
    {
        tensor<T, I> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = a_x_0[i] * a_x_1[i];

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
        requires (sizeof...(J) > 0)
    tensor<T, I, J ...> multiply(
        const tensor<T, I, J ...>& a_x_0,
        const tensor<T, I, J ...>& a_x_1
    )
    {
        tensor<T, I, J ...> l_result;

        for (int i = 0; i < I; i++)
            l_result[i] = multiply(a_x_0[i], a_x_1[i]);

        return l_result;
        
    }

    template<typename T, size_t I>
    T average(
        const tensor<T, I>& a_x
    )
    {
        return additive_aggregate(a_x) / T(I);
    }

    template<typename T, size_t I, size_t ... J>
        requires (sizeof...(J) > 0)
    tensor<T, J ...> average(
        const tensor<T, I, J ...>& a_x
    )
    {
        return multiply(additive_aggregate(a_x), T(1.0 / double(I)));
    }

    /// @brief This function currently flips the tensor's outermost two ranks.
    /// @tparam I 
    /// @tparam J 
    /// @tparam K 
    /// @tparam ...L 
    /// @param a_tensor 
    /// @return 
    template<typename T, size_t I, size_t J, size_t ... K>
    tensor<T, J, I, K ...> transpose(
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
    tensor<T, I, J ...> negate(
        const tensor<T, I, J ...>& a_x
    )
    {
        tensor<T, I, J ...> l_result = multiply(a_x, T(-1.0));
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

template<typename T, size_t I, size_t J>
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I, J>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element << '\n';

    a_ostream << '\n';

    return a_ostream;

}

template<typename T, size_t I, size_t ... J>
std::ostream& operator<<(std::ostream& a_ostream, const aurora::tensor<T, I, J ...>& a_tensor)
{
    for (const auto& l_element : a_tensor)
        a_ostream << l_element;

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
