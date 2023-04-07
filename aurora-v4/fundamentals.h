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
