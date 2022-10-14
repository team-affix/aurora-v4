#pragma once
#include "affix-base/pch.h"
#include "fundamentals.h"

namespace aurora
{
	namespace oneshot
	{
		class parameter_vector : public std::vector<double>
		{
		private:
			size_t m_next_index = 0;
			std::uniform_real_distribution<double> m_uniform_real_distribution;

		public:
			parameter_vector(
				const double& a_minimum_parameter_value,
				const double& a_maximum_parameter_value
			) :
				m_uniform_real_distribution(a_minimum_parameter_value, a_maximum_parameter_value)
			{

			}

			parameter_vector& operator=(
				const std::vector<double>& a_vector
				)
			{
				assert(size() == a_vector.size());
				for (int i = 0; i < size(); i++)
					at(i) = a_vector[i];
				return *this;
			}

		public:
			size_t next_index(

			)
			{
				return m_next_index;
			}

			void next_index(
				const size_t& a_next_index
			)
			{
				m_next_index = a_next_index;
			}

			double next(

			)
			{
				if (m_next_index == size())
				{
					push_back(m_uniform_real_distribution(i_default_random_engine));
					m_next_index++;
					return back();
				}
				else if (m_next_index < size())
				{
					double l_result = at(m_next_index);
					m_next_index++;
					return l_result;
				}
				else
				{
					throw std::exception("Error: m_next_index was larger than the size of the parameter vector.");
				}
			}

			std::vector<double> next(
				const size_t& a_size
			)
			{
				std::vector<double> l_result(a_size);
				for (int i = 0; i < a_size; i++)
					l_result[i] = next();
				return l_result;
			}

			std::vector<std::vector<double>> next(
				const size_t& a_rows,
				const size_t& a_cols
			)
			{
				std::vector<std::vector<double>> l_result(a_rows);
				for (int i = 0; i < a_rows; i++)
					l_result[i] = next(a_cols);
				return l_result;
			}

			std::vector<std::vector<std::vector<double>>> next(
				const size_t& a_depth,
				const size_t& a_rows,
				const size_t& a_cols
			)
			{
				std::vector<std::vector<std::vector<double>>> l_result(a_depth);
				for (int i = 0; i < a_depth; i++)
					l_result[i] = next(a_rows, a_cols);
				return l_result;
			}

		};

		inline double sigmoid(
			const double& a_x
		)
		{
			return 1.0 / (1.0 + std::exp(-a_x));
		}

		inline double leaky_relu(
			const double& a_x,
			const double& a_m
		)
		{
			if (a_x >= 0)
				return a_x;
			else
				return a_m * a_x;
		}

		std::vector<double> range(
			const std::vector<double>& a_x,
			const size_t& a_start_index,
			const size_t& a_size
		)
		{
			std::vector<double> l_result(a_x.begin() + a_start_index, a_x.begin() + a_start_index + a_size);
			return l_result;
		}

		inline std::vector<std::vector<double>> range(
			const std::vector<std::vector<double>>& a_matrix,
			const size_t& a_top_index,
			const size_t& a_left_index,
			const size_t& a_height,
			const size_t& a_width
		)
		{
			std::vector<std::vector<double>> l_result(a_height);

			for (int i = 0; i < a_height; i++)
			{
				l_result[i] = range(a_matrix[a_top_index + i], a_left_index, a_width);
			}

			return l_result;

		}

		inline std::vector<std::vector<std::vector<double>>> range(
			const std::vector<std::vector<std::vector<double>>>& a_tensor,
			const size_t& a_front_index,
			const size_t& a_top_index,
			const size_t& a_left_index,
			const size_t& a_depth,
			const size_t& a_height,
			const size_t& a_width
		)
		{
			std::vector<std::vector<std::vector<double>>> l_result(a_depth);

			for (int i = 0; i < a_depth; i++)
			{
				l_result[i] = range(a_tensor[a_front_index + i], a_top_index, a_left_index, a_height, a_width);
			}

			return l_result;

		}

		inline double additive_aggregate(
			const std::vector<double>& a_x
		)
		{
			assert(a_x.size() > 0);

			double l_result = 0;

			for (auto& l_element : a_x)
				l_result += l_element;

			return l_result;

		}

		inline std::vector<double> add(
			const std::vector<double>& a_x_0,
			const std::vector<double>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<double> l_result(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_result[i] = a_x_0[i] + a_x_1[i];
			}

			return l_result;

		}

		inline std::vector<std::vector<double>> add(
			const std::vector<std::vector<double>>& a_x_0,
			const std::vector<std::vector<double>>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<std::vector<double>> l_result(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_result[i] = add(a_x_0[i], a_x_1[i]);
			}

			return l_result;

		}

		inline std::vector<std::vector<std::vector<double>>> add(
			const std::vector<std::vector<std::vector<double>>>& a_x_0,
			const std::vector<std::vector<std::vector<double>>>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<std::vector<std::vector<double>>> l_result(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_result[i] = add(a_x_0[i], a_x_1[i]);
			}

			return l_result;

		}

		inline std::vector<double> subtract(
			const std::vector<double>& a_x_0,
			const std::vector<double>& a_x_1
		)
		{
			std::vector<double> l_y(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_y[i] = a_x_0[i] - a_x_1[i];
			}

			return l_y;

		}

		inline std::vector<std::vector<double>> subtract(
			const std::vector<std::vector<double>>& a_x_0,
			const std::vector<std::vector<double>>& a_x_1
		)
		{
			std::vector<std::vector<double>> l_y(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_y[i] = subtract(a_x_0[i], a_x_1[i]);
			}

			return l_y;

		}

		inline std::vector<std::vector<std::vector<double>>> subtract(
			const std::vector<std::vector<std::vector<double>>>& a_x_0,
			const std::vector<std::vector<std::vector<double>>>& a_x_1
		)
		{
			std::vector<std::vector<std::vector<double>>> l_y(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_y[i] = subtract(a_x_0[i], a_x_1[i]);
			}

			return l_y;

		}

		inline std::vector<double> additive_aggregate(
			const std::vector<std::vector<double>>& a_x
		)
		{
			std::vector<double> l_result = a_x[0];
			for (int i = 1; i < a_x.size(); i++)
				l_result = add(l_result, a_x[i]);
			return l_result;
		}

		inline double average(
			const std::vector<double>& a_x
		)
		{
			return additive_aggregate(a_x) / (double)a_x.size();
		}

		inline std::vector<std::vector<double>> transpose(
			const std::vector<std::vector<double>>& a_x
		)
		{
			std::vector<std::vector<double>> l_result;

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
					// Send the value to the correct location
					l_result[j][i] = a_x[i][j];
				}
			}

			return l_result;

		}

		inline std::vector<double> hadamard(
			const std::vector<double>& a_x_0,
			const std::vector<double>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<double> l_y(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_y[i] = a_x_0[i] * a_x_1[i];
			}

			return l_y;

		}

		inline std::vector<std::vector<double>> hadamard(
			const std::vector<std::vector<double>>& a_x_0,
			const std::vector<std::vector<double>>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<std::vector<double>> l_y(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_y[i] = hadamard(a_x_0[i], a_x_1[i]);
			}

			return l_y;

		}

		inline std::vector<std::vector<std::vector<double>>> hadamard(
			const std::vector<std::vector<std::vector<double>>>& a_x_0,
			const std::vector<std::vector<std::vector<double>>>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<std::vector<std::vector<double>>> l_y(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_y[i] = hadamard(a_x_0[i], a_x_1[i]);
			}

			return l_y;

		}

		inline double multiply(
			const std::vector<double>& a_x_0,
			const std::vector<double>& a_x_1
		)
		{
			assert(a_x_0.size() == a_x_1.size());

			std::vector<double> l_multiply_ys(a_x_0.size());

			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_multiply_ys[i] = a_x_0[i] * a_x_1[i];
			}

			return additive_aggregate(l_multiply_ys);

		}

		inline std::vector<double> multiply(
			const std::vector<double>& a_x_0,
			double a_x_1
		)
		{
			std::vector<double> l_result(a_x_0.size());
			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_result[i] = a_x_0[i] * a_x_1;
			}
			return l_result;
		}

		inline std::vector<double> multiply(
			const std::vector<std::vector<double>>& a_x_0,
			const std::vector<double>& a_x_1
		)
		{
			assert(a_x_0[0].size() == a_x_1.size());
			auto l_transpose = transpose(a_x_0);
			std::vector<std::vector<double>> l_scaled_transpose(l_transpose.size());
			for (int i = 0; i < a_x_1.size(); i++)
			{
				l_scaled_transpose[i] = multiply(l_transpose[i], a_x_1[i]);
			}
			return additive_aggregate(l_scaled_transpose);
		}

		inline std::vector<std::vector<double>> multiply(
			const std::vector<std::vector<double>>& a_x_0,
			double a_x_1
		)
		{
			std::vector<std::vector<double>> l_result(a_x_0.size());
			for (int i = 0; i < a_x_0.size(); i++)
			{
				l_result[i] = multiply(a_x_0[i], a_x_1);
			}
			return l_result;
		}

		inline std::vector<std::vector<double>> multiply(
			const std::vector<std::vector<double>>& a_x_0,
			const std::vector<std::vector<double>>& a_x_1
		)
		{
			std::vector<std::vector<double>> l_result(a_x_0.size());
			std::vector<std::vector<double>> l_x_1_transpose = transpose(a_x_1);
			for (int i = 0; i < a_x_0.size(); i++)
			{
				std::vector<double> l_result_row(a_x_1[0].size());
				for (int j = 0; j < l_x_1_transpose.size(); j++)
				{
					l_result_row[j] = multiply(a_x_0[i], l_x_1_transpose[j]);
				}
				l_result[i] = l_result_row;
			}
			return l_result;
		}

		inline std::vector<double> sigmoid(
			const std::vector<double>& a_x
		)
		{
			std::vector<double> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = sigmoid(a_x[i]);
			return l_result;
		}

		inline std::vector<double> tanh(
			const std::vector<double>& a_x
		)
		{
			std::vector<double> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = std::tanh(a_x[i]);
			return l_result;
		}

		inline std::vector<double> leaky_relu(
			const std::vector<double>& a_x,
			const double& a_m
		)
		{
			std::vector<double> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = leaky_relu(a_x[i], a_m);
			return l_result;
		}

		inline std::vector<std::vector<double>> sigmoid(
			const std::vector<std::vector<double>>& a_x
		)
		{
			std::vector<std::vector<double>> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = sigmoid(a_x[i]);
			return l_result;
		}

		inline std::vector<std::vector<double>> tanh(
			const std::vector<std::vector<double>>& a_x
		)
		{
			std::vector<std::vector<double>> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = tanh(a_x[i]);
			return l_result;
		}

		inline std::vector<std::vector<double>> leaky_relu(
			const std::vector<std::vector<double>>& a_x,
			const double& a_m
		)
		{
			std::vector<std::vector<double>> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = leaky_relu(a_x[i], a_m);
			return l_result;
		}

		inline std::vector<std::vector<std::vector<double>>> sigmoid(
			const std::vector<std::vector<std::vector<double>>>& a_x
		)
		{
			std::vector<std::vector<std::vector<double>>> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = sigmoid(a_x[i]);
			return l_result;
		}

		inline std::vector<std::vector<std::vector<double>>> tanh(
			const std::vector<std::vector<std::vector<double>>>& a_x
		)
		{
			std::vector<std::vector<std::vector<double>>> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = tanh(a_x[i]);
			return l_result;
		}

		inline std::vector<std::vector<std::vector<double>>> leaky_relu(
			const std::vector<std::vector<std::vector<double>>>& a_x,
			const double& a_m
		)
		{
			std::vector<std::vector<std::vector<double>>> l_result(a_x.size());
			for (int i = 0; i < a_x.size(); i++)
				l_result[i] = leaky_relu(a_x[i], a_m);
			return l_result;
		}

		inline double magnitude(
			const std::vector<double>& a_x
		)
		{
			double l_sum = 0;
			for (int i = 0; i < a_x.size(); i++)
				l_sum += a_x[i] * a_x[i];
			return std::sqrt(l_sum);
		}

		inline std::vector<double> normalize(
			const std::vector<double>& a_x
		)
		{
			std::vector<double> l_result(a_x.size());

			double l_magnitude = magnitude(a_x);

			for (int i = 0; i < a_x.size(); i++)
			{
				l_result[i] = a_x[i] / l_magnitude;
			}

			return l_result;

		}

		inline std::vector<double> concat(
			const std::vector<double>& a_x_0,
			const std::vector<double>& a_x_1
		)
		{
			std::vector<double> l_result(a_x_0.size() + a_x_1.size());
			for (int i = 0; i < a_x_0.size(); i++)
				l_result[i] = a_x_0[i];
			for (int i = 0; i < a_x_1.size(); i++)
				l_result[a_x_0.size() + i] = a_x_1[i];
			return l_result;
		}

		inline void lstm_timestep(
			const std::vector<double>& a_x,
			std::vector<double>& a_hx,
			std::vector<double>& a_cx,
			const std::vector<std::vector<double>>& a_forget_gate_weights,
			const std::vector<std::vector<double>>& a_input_limit_gate_weights,
			const std::vector<std::vector<double>>& a_input_gate_weights,
			const std::vector<std::vector<double>>& a_output_gate_weights,
			const std::vector<double>& a_forget_gate_bias,
			const std::vector<double>& a_input_limit_gate_bias,
			const std::vector<double>& a_input_gate_bias,
			const std::vector<double>& a_output_gate_bias
		)
		{
			// Concatenate x and hx
			std::vector<double> l_x_hx_concat = concat(a_x, a_hx);

			// Calculate forget gate y
			std::vector<double> l_forget_gate_y = multiply(a_forget_gate_weights, l_x_hx_concat);
			l_forget_gate_y = add(l_forget_gate_y, a_forget_gate_bias);
			l_forget_gate_y = sigmoid(l_forget_gate_y);

			// Forget parts of cell state
			a_cx = hadamard(a_cx, l_forget_gate_y);

			// Calculate input limit gate y
			std::vector<double> l_input_limit_gate_y = multiply(a_input_limit_gate_weights, l_x_hx_concat);
			l_input_limit_gate_y = add(l_input_limit_gate_y, a_input_limit_gate_bias);
			l_input_limit_gate_y = sigmoid(l_input_limit_gate_y);

			// Calculate input gate y
			std::vector<double> l_input_gate_y = multiply(a_input_gate_weights, l_x_hx_concat);
			l_input_gate_y = add(l_input_gate_y, a_input_gate_bias);
			l_input_gate_y = tanh(l_input_gate_y);

			// Calculate input to cell state
			std::vector<double> l_input_to_cell_state = hadamard(l_input_gate_y, l_input_limit_gate_y);

			// Input to the cell state
			a_cx = add(a_cx, l_input_to_cell_state);

			// Calculate output gate y
			std::vector<double> l_output_gate_y = multiply(a_output_gate_weights, l_x_hx_concat);
			l_output_gate_y = add(l_output_gate_y, a_output_gate_bias);
			l_output_gate_y = sigmoid(l_output_gate_y);

			// Compute output to lstm timestep
			a_hx = hadamard(tanh(a_cx), l_output_gate_y);

		}

		inline double mean_squared_error(
			const double& a_prediction,
			const double& a_desired
		)
		{
			double l_error = a_prediction - a_desired;
			return l_error * l_error;
		}

		inline double mean_squared_error(
			const std::vector<double>& a_prediction,
			const std::vector<double>& a_desired
		)
		{
			double l_sum = 0;
			for (int i = 0; i < a_prediction.size(); i++)
			{
				double l_error = a_prediction[i] - a_desired[i];
				l_sum += l_error * l_error;
			}
			return l_sum / (double)a_prediction.size();
		}

		inline double mean_squared_error(
			const std::vector<std::vector<double>>& a_prediction,
			const std::vector<std::vector<double>>& a_desired
		)
		{
			double l_sum = 0;
			for (int i = 0; i < a_prediction.size(); i++)
			{
				for (int j = 0; j < a_prediction[0].size(); j++)
				{
					double l_error = a_prediction[i][j] - a_desired[i][j];
					l_sum += l_error * l_error;
				}
			}
			return l_sum / (double)a_prediction.size() / (double)a_prediction[0].size();
		}

		inline double random(
			const double& a_minimum,
			const double& a_maximum
		)
		{
			std::uniform_real_distribution<double> l_urd(a_minimum, a_maximum);
			return l_urd(i_default_random_engine);
		}

		inline std::vector<double> make(
			const size_t& a_size
		)
		{
			std::vector<double> l_result(a_size);
			return l_result;
		}

		inline std::vector<std::vector<double>> make(
			const size_t& a_rows,
			const size_t& a_cols
		)
		{
			std::vector<std::vector<double>> l_result(a_rows);
			for (int i = 0; i < l_result.size(); i++)
				l_result[i] = make(a_cols);
			return l_result;
		}

		inline std::vector<double> random(
			const size_t& a_size,
			const double& a_minimum,
			const double& a_maximum
		)
		{
			std::uniform_real_distribution<double> l_urd(a_minimum, a_maximum);
			std::vector<double> l_result(a_size);
			for (int i = 0; i < a_size; i++)
				l_result[i] = l_urd(i_default_random_engine);
			return l_result;
		}

		inline std::vector<std::vector<double>> random(
			const size_t& a_rows,
			const size_t& a_cols,
			const double& a_minimum,
			const double& a_maximum
		)
		{
			std::uniform_real_distribution<double> l_urd(a_minimum, a_maximum);
			std::vector<std::vector<double>> l_result = make(a_rows, a_cols);
			for (int i = 0; i < a_rows; i++)
				for (int j = 0; j < a_cols; j++)
					l_result[i][j] = l_urd(i_default_random_engine);
			return l_result;
		}

		inline std::vector<double> flatten(
			const std::vector<std::vector<double>>& a_x
		)
		{
			std::vector<double> l_result(a_x.size() * a_x[0].size());

			for (int i = 0; i < a_x.size(); i++)
			{
				for (int j = 0; j < a_x[0].size(); j++)
				{
					l_result[i * a_x[0].size() + j] = a_x[i][j];
				}
			}

			return l_result;

		}

		inline std::vector<double> flatten(
			const std::vector<std::vector<std::vector<double>>>& a_x
		)
		{
			std::vector<double> l_result(a_x.size() * a_x[0].size() * a_x[0][0].size());

			for (int i = 0; i < a_x.size(); i++)
			{
				for (int j = 0; j < a_x[0].size(); j++)
				{
					for (int k = 0; k < a_x[0][0].size(); k++)
					{
						l_result[i * a_x[0].size() + j * a_x[0][0].size() + k] = a_x[i][j][k];
					}
				}
			}

			return l_result;

		}

		inline std::vector<double> convolve(
			const std::vector<std::vector<double>>& a_x,
			const std::vector<std::vector<double>>& a_filter,
			const size_t& a_stride
		)
		{
			// Since the first dimension is considered to be height of the filter, we reserve the first dimension as
			// being non-spacial, and hence we use only the [0].size() as the width of our matrix.

			int l_right_most_position = a_x[0].size() - a_filter[0].size();

			assert(l_right_most_position >= 0);

			size_t l_convolution_count = (l_right_most_position / a_stride) + 1;

			std::vector<double> l_result(l_convolution_count);

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

		inline std::vector<std::vector<double>> convolve(
			const std::vector<std::vector<std::vector<double>>>& a_x,
			const std::vector<std::vector<std::vector<double>>>& a_filter,
			const size_t& a_stride
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

			std::vector<std::vector<double>> l_result(l_vertical_convolution_count);

			for (int i = 0; i < l_vertical_convolution_count; i++)
			{
				std::vector<double> l_result_row(l_horizontal_convolution_count);
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

		inline std::vector<double> average_pool(
			const std::vector<double>& a_x,
			const size_t& a_bin_width,
			const size_t& a_stride = 1
		)
		{
			size_t l_right_most_index = a_x.size() - a_bin_width;

			size_t l_pool_size = (l_right_most_index / a_stride) + 1;

			std::vector<double> l_result(l_pool_size);

			for (int i = 0; i < l_pool_size; i++)
			{
				l_result[i] = average(range(a_x, i * a_stride, a_bin_width));
			}

			return l_result;

		}

		inline std::vector<std::vector<double>> average_pool(
			const std::vector<std::vector<double>>& a_x,
			const size_t& a_bin_height,
			const size_t& a_bin_width,
			const size_t& a_stride = 1
		)
		{
			size_t l_top_most_index = a_x.size() - a_bin_height;
			size_t l_right_most_index = a_x[0].size() - a_bin_width;

			size_t l_pool_height = (l_top_most_index / a_stride) + 1;
			size_t l_pool_width = (l_right_most_index / a_stride) + 1;

			std::vector<std::vector<double>> l_result(l_pool_height);

			for (int i = 0; i < l_pool_height; i++)
			{
				std::vector<double> l_result_row(l_pool_width);

				for (int j = 0; j < l_pool_width; j++)
				{
					l_result_row[j] = average(flatten(range(a_x, i * a_stride, j * a_stride, a_bin_height, a_bin_width)));
				}

				l_result[i] = l_result_row;

			}

			return l_result;

		}

		class particle_optimizer
		{
		public:
			parameter_vector& m_position;
			std::vector<double> m_best_position;
			std::vector<double> m_velocity;
			double m_best_reward = -INFINITY;

		public:
			particle_optimizer(
				parameter_vector& a_parameter_vector
			) :
				m_position(a_parameter_vector),
				m_best_position(a_parameter_vector.size()),
				m_velocity(a_parameter_vector.size())
			{

			}

		public:
			void update(
				const double& a_w,
				const double& a_c1,
				const double& a_c2,
				const double& a_reward,
				const std::vector<double>& a_global_best_position
			)
			{
				if (a_reward > m_best_reward)
				{
					m_best_position = m_position;
					m_best_reward = a_reward;
				}
				std::vector<double> l_weighted_particle_velocity = oneshot::multiply(m_velocity, a_w);
				std::vector<double> l_cognitive_term = oneshot::multiply(oneshot::multiply(oneshot::subtract(m_best_position, m_position), a_c1), oneshot::random(0, 1));
				std::vector<double> l_social_term = oneshot::multiply(oneshot::multiply(oneshot::subtract(a_global_best_position, m_position), a_c2), oneshot::random(0, 1));
				m_velocity = oneshot::add(oneshot::add(l_weighted_particle_velocity, l_cognitive_term), l_social_term);
				m_position = oneshot::add(m_position, m_velocity);
			}

		};

		class particle_swarm_optimizer
		{
		private:
			std::vector<particle_optimizer> m_particle_optimizers;
			double m_w = 0;
			double m_c1 = 0;
			double m_c2 = 0;
			double m_global_best_reward = -INFINITY;
			std::vector<double> m_global_best_position;

		public:
			particle_swarm_optimizer(
				const std::vector<particle_optimizer>& a_particle_optimizers,
				const double& a_w,
				const double& a_c1,
				const double& a_c2
			) :
				m_particle_optimizers(a_particle_optimizers),
				m_w(a_w),
				m_c1(a_c1),
				m_c2(a_c2)
			{

			}

			void update(
				const std::vector<double>& a_particle_rewards
			)
			{
				// Get the global best position if it has improved
				for (int i = 0; i < a_particle_rewards.size(); i++)
				{
					if (a_particle_rewards[i] > m_global_best_reward)
					{
						m_global_best_reward = a_particle_rewards[i];
						m_global_best_position = m_particle_optimizers[i].m_position;
					}
				}
				// Update all particle positions
				for (int i = 0; i < m_particle_optimizers.size(); i++)
				{
					m_particle_optimizers[i].update(m_w, m_c1, m_c2, a_particle_rewards[i], m_global_best_position);
				}
			}

			double global_best_reward(

			)
			{
				return m_global_best_reward;
			}

			std::vector<double> global_best_position(

			)
			{
				return m_global_best_position;
			}

		};

	}

}
