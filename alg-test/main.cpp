#include "aurora-v4/aurora.h"
#include <assert.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>
#include <cmath>
#include <set>

using namespace aurora;

// Returns the number of milliseconds elapsed since the start.
long long duration_ms(
    const std::chrono::high_resolution_clock::time_point& a_start
)
{
    return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - a_start)).count();
}

namespace aurora
{
    template<>
    inline size_t constant<size_t>(
        const double& a_val
    )
    {
        return (size_t)a_val;
    }
}

void tnn_test(

)
{
    std::cout << "Testing TNN" << std::endl;

    constexpr size_t CONCURRENT_INSTANCES = 4;
    constexpr std::array<size_t, 3> INSTANCE_DIMENSIONS = {2, 5, 1};
    
    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };
    
    model::begin();

	auto l_x = input<CONCURRENT_INSTANCES, INSTANCE_DIMENSIONS.front()>();
	auto l_desired_y = input<CONCURRENT_INSTANCES, INSTANCE_DIMENSIONS.back()>();

	latent_tensor<CONCURRENT_INSTANCES, INSTANCE_DIMENSIONS.back()> l_y;

    auto l_w0 = input<INSTANCE_DIMENSIONS[1], INSTANCE_DIMENSIONS[0]>(l_randomly_generate_parameter);
    auto l_w1 = input<INSTANCE_DIMENSIONS[2], INSTANCE_DIMENSIONS[1]>(l_randomly_generate_parameter);
    auto l_b1 = input<INSTANCE_DIMENSIONS[1]>(l_randomly_generate_parameter);
    auto l_b2 = input<INSTANCE_DIMENSIONS[2]>(l_randomly_generate_parameter);

	for (int i = 0; i < l_x.size(); i++)
	{
        auto l_w0_y = multiply(l_w0, l_x[i]);
        auto l_b1_y = add(l_w0_y, l_b1);
        auto l_leaky_relu_0 = leaky_relu(l_b1_y, 0.3);
        auto l_w1_y = multiply(l_w1, l_leaky_relu_0);
        auto l_b2_y = add(l_w1_y, l_b2);
        auto l_sigmoid = sigmoid(l_b2_y);
        l_y[i] = l_sigmoid;
	}

    auto l_mse_loss = mean_squared_error(l_y, l_desired_y)->depend();

    model l_model = model::end();

	const int CHECKPOINT = 100000;

    std::stringstream l_x_ss(
        "0 0\n"
        "0 1\n"
        "1 0\n"
        "1 1\n"
    );

    std::stringstream l_y_ss(
        "0\n"
        "1\n"
        "1\n"
        "0\n"
    );

	tensor<double, CONCURRENT_INSTANCES, INSTANCE_DIMENSIONS.front()> l_ts_x;
	tensor<double, CONCURRENT_INSTANCES, INSTANCE_DIMENSIONS.back()>  l_ts_y;

    l_x_ss >> l_ts_x;
    l_y_ss >> l_ts_y;

    gradient_descent_with_momentum l_optimizer(flatten(l_w0, l_w1, l_b1, l_b2), false, 0.002, 0.9);

    std::chrono::time_point l_start = std::chrono::high_resolution_clock::now();

	for (int epoch = 0; epoch < 500000; epoch++)
	{
		set_state(l_x, l_ts_x);
		set_state(l_desired_y, l_ts_y);

		l_model.fwd();

		l_mse_loss.m_partial_gradient = 1;

		l_model.bwd();

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[0][0]->m_state << std::endl;

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[1][0]->m_state << std::endl;

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[2][0]->m_state << std::endl;

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[3][0]->m_state << std::endl;

		if (epoch % CHECKPOINT == 0)
			std::cout << std::endl;

        l_optimizer.update();

	}

    l_model.fwd();

    assert(l_y[0][0]->m_state < 0.07);
    assert(l_y[1][0]->m_state > 0.9);
    assert(l_y[2][0]->m_state > 0.9);
    assert(l_y[3][0]->m_state < 0.07);

	std::cout
        << std::endl << "PERIOD OF TRAINING (ms): "
        << duration_ms(l_start)
        << std::endl;

}

// double sign_d(const double& a_double)
// {
// 	if (a_double >= 0)
// 		return 1.0;
// 	else
// 		return -1.0;
// }

void parabola_test(

)
{
    std::cout << "Testing parabola fit using TNN" << std::endl;

    constexpr size_t CONCURRENT_INSTANCES = 4;
    constexpr std::array<size_t, 3> LAYER_DIMS = {1, 20, 1};
    
    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

    model::begin();
	
    auto l_x = input<CONCURRENT_INSTANCES, LAYER_DIMS.front()>();
    
    latent_tensor<CONCURRENT_INSTANCES, LAYER_DIMS.back()> l_y;

    auto l_w0 = input<LAYER_DIMS[1], LAYER_DIMS[0]>(l_randomly_generate_parameter);
    auto l_w1 = input<LAYER_DIMS[2], LAYER_DIMS[1]>(l_randomly_generate_parameter);
    auto l_b1 = input<LAYER_DIMS[1]>(l_randomly_generate_parameter);
    auto l_b2 = input<LAYER_DIMS[2]>(l_randomly_generate_parameter);

	for (int i = 0; i < CONCURRENT_INSTANCES; i++)
	{
        auto l_w0_y = multiply(l_w0, l_x[i]);
        auto l_b1_y = add(l_b1, l_w0_y);
        auto l_leaky_relu_0 = leaky_relu(l_b1_y, 0.3);
        auto l_w1_y = multiply(l_w1, l_leaky_relu_0);
        auto l_b2_y = add(l_b2, l_w1_y);
        auto l_leaky_relu_1 = leaky_relu(l_b2_y, 0.3);
        l_y[i] = l_leaky_relu_1;
	}

	auto l_desired_y = input<CONCURRENT_INSTANCES, LAYER_DIMS.back()>();

	auto l_error = mean_squared_error(l_y, l_desired_y)->depend();

    model l_model = model::end();

    gradient_descent_with_momentum l_optimizer(flatten(l_w0, l_w1, l_b1, l_b2), true, 0.02, 0.9);

	double l_cost_momentum = 0;

	const size_t CHECKPOINT_INTERVAL = 10000;

    std::uniform_real_distribution<double> l_ts_urd(-10, 10);

	for (int epoch = 0; epoch < CHECKPOINT_INTERVAL * 30; epoch++)
	{
		double l_cost = 0;

		tensor<double, CONCURRENT_INSTANCES, LAYER_DIMS.front()> l_ts_x;
		tensor<double, CONCURRENT_INSTANCES, LAYER_DIMS.back()> l_ts_y;

		for (int i = 0; i < CONCURRENT_INSTANCES; i++)
		{
            l_ts_x[i][0] = l_ts_urd(l_dre);
            l_ts_y[i][0] = l_ts_x[i][0] * l_ts_x[i][0];
		}

		set_state(l_x, l_ts_x);
		set_state(l_desired_y, l_ts_y);

		l_model.fwd();

		l_error.m_partial_gradient = 1;
		l_cost = l_error.m_state;

		l_model.bwd();

		if (epoch % CHECKPOINT_INTERVAL == 0)
		{
			for (int i = 0; i < l_y.size(); i++)
			{
				std::cout << "INPUT: " << l_ts_x[i][0] << ", PREDICTION: " << l_y[i][0]->m_state << ", DESIRED: " << l_ts_y[i][0] << std::endl;
			}
		}


		l_cost_momentum = 0.99 * l_cost_momentum + 0.01 * l_cost;

        l_optimizer.update();

		if (epoch % CHECKPOINT_INTERVAL == 0)
			std::cout << "    LOSS FOR ABOVE EPOCH: " << l_cost_momentum << std::endl;

	}

    assert(l_cost_momentum < 1.0);

}

void lstm_test(

)
{
    std::cout << "Testing LSTM" << std::endl;

    constexpr size_t CONCURRENT_INSTANCES = 2;
    constexpr size_t LSTM_TIMESTEPS = 4;
    constexpr std::array<size_t, 2> LSTM_DIMS = {2, 10};
    constexpr std::array<size_t, 3> TNN_DIMS = {LSTM_DIMS.back(), 5, 1};

    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

    model::begin();

    auto l_x = input<CONCURRENT_INSTANCES, LSTM_TIMESTEPS, LSTM_DIMS.front()>();

    latent_tensor<CONCURRENT_INSTANCES, LSTM_TIMESTEPS, TNN_DIMS.back()> l_y;

    auto l_cx = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
    auto l_hx = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);

    auto l_forget_gate_bias = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
    auto l_input_limit_gate_bias = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
    auto l_input_gate_bias = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
    auto l_output_gate_bias = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
    
    auto l_forget_gate_weights = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);
    auto l_input_limit_gate_weights = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);
    auto l_input_gate_weights = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);
    auto l_output_gate_weights = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);

    auto l_w0 = input<TNN_DIMS[1], TNN_DIMS[0]>(l_randomly_generate_parameter);
    auto l_b1 = input<TNN_DIMS[1]>(l_randomly_generate_parameter);
    auto l_w1 = input<TNN_DIMS[2], TNN_DIMS[1]>(l_randomly_generate_parameter);
    auto l_b2 = input<TNN_DIMS[2]>(l_randomly_generate_parameter);

	for (int i = 0; i < CONCURRENT_INSTANCES; i++)
	{
		auto l_lstm_y = lstm(
            l_x[i],
            l_cx,
            l_hx,
            l_forget_gate_bias,
            l_input_limit_gate_bias,
            l_input_gate_bias,
            l_output_gate_bias,
            l_forget_gate_weights,
            l_input_limit_gate_weights,
            l_input_gate_weights,
            l_output_gate_weights
        );

        latent_tensor<LSTM_TIMESTEPS, TNN_DIMS.back()> l_tnn_ys;

		for (int j = 0; j < LSTM_TIMESTEPS; j++)
		{
            auto l_layer_0_y = leaky_relu(add(multiply(l_w0, l_lstm_y[j]), l_b1), 0.3);
            auto l_layer_1_y = leaky_relu(add(multiply(l_w1, l_layer_0_y), l_b2), 0.3);
            l_tnn_ys[j] = l_layer_1_y;
		}

		l_y[i] = l_tnn_ys;

	}

	auto l_desired_y = input<CONCURRENT_INSTANCES, LSTM_TIMESTEPS, TNN_DIMS.back()>();

	auto l_error = mean_squared_error(l_y, l_desired_y)->depend();

    model l_model = model::end();

    auto l_param_concat = flatten(
        l_cx,
        l_hx,
        l_forget_gate_bias,
        l_input_limit_gate_bias,
        l_input_gate_bias,
        l_output_gate_bias,
        l_forget_gate_weights,
        l_input_limit_gate_weights,
        l_input_gate_weights,
        l_output_gate_weights,
        l_w0,
        l_b1,
        l_w1,
        l_b2
    );

	gradient_descent_with_momentum l_optimizer(l_param_concat, true, 0.02, 0.9);

    std::stringstream l_ts_x(
        "0 0\n"
        "0 1\n"
        "1 0\n"
        "1 1\n"
        "0 1\n"
        "0 1\n"
        "1 0\n"
        "1 1\n"
    );

    std::stringstream l_ts_y(
        "0\n"
        "1\n"
        "1\n"
        "0\n"
        "0\n"
        "1\n"
        "1\n"
        "1\n"
    );

	tensor<double, CONCURRENT_INSTANCES, LSTM_TIMESTEPS, LSTM_DIMS.front()> l_training_set_xs;
	tensor<double, CONCURRENT_INSTANCES, LSTM_TIMESTEPS, TNN_DIMS.back()>   l_training_set_ys;

    l_ts_x >> l_training_set_xs;
    l_ts_y >> l_training_set_ys;

	const size_t CHECKPOINT = 1000;

	for (int epoch = 0; epoch < 5 * CHECKPOINT; epoch++)
	{
		double l_cost = 0;

		set_state(l_x, l_training_set_xs);
		set_state(l_desired_y, l_training_set_ys);

		// Carry forward
		l_model.fwd();

		// Signal output
		l_cost += l_error.m_state;
		l_error.m_partial_gradient = 1;

		// Carry backward
		l_model.bwd();

		if (epoch % CHECKPOINT == 0)
		{
			for (int i = 0; i < l_y.size(); i++)
				std::cout << "PREDICTION: \n" << get_state(l_y) << std::endl;
			std::cout << std::endl;
		}

		l_optimizer.update();

		if (epoch % CHECKPOINT == 0)
			std::cout << "COST: " << l_cost << std::endl << std::endl;

	}

    assert(l_error.m_state < 0.001);

}

// void lstm_stacked_test(

// )
// {
//     constexpr size_t LSTM_TIMESTEPS = 4;
//     constexpr std::array<size_t, 4> STACK_DIMS = { 2, 20, 20, 1 };

//     constexpr size_t LSTM_PARAM_COUNT = ;

//     std::mt19937 l_dre(26);
//     std::uniform_real_distribution<double> l_urd(-1, 1);
    
//     std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

//     auto l_x = input<LSTM_TIMESTEPS, STACK_DIMS.front()>();

//     auto l_cx_0 = input<STACK_DIMS[1]>(l_randomly_generate_parameter);
//     auto l_hx_0 = input<STACK_DIMS[1]>(l_randomly_generate_parameter);

//     auto l_forget_gate_bias_0 = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
//     auto l_input_limit_gate_bias_0 = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
//     auto l_input_gate_bias_0 = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
//     auto l_output_gate_bias_0 = input<LSTM_DIMS.back()>(l_randomly_generate_parameter);
    
//     auto l_forget_gate_weights_0 = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);
//     auto l_input_limit_gate_weights_0 = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);
//     auto l_input_gate_weights_0 = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);
//     auto l_output_gate_weights_0 = input<LSTM_DIMS.back(), LSTM_DIMS.back() + LSTM_DIMS.front()>(l_randomly_generate_parameter);

// 	auto l_lstm_0 = l_model.lstm(pointers(l_x), 20);
// 	auto l_lstm_1 = l_model.lstm(l_lstm_0, 20);
// 	auto l_lstm_2 = l_model.lstm(l_lstm_1, 1);

// 	sgp_ptr_matrix l_y;

// 	for (int i = 0; i < l_lstm_2.size(); i++)
// 	{
// 		auto l_tnn_y = l_lstm_2[i];
// 		l_tnn_y = l_model.weight_junction(l_tnn_y, 5);
// 		l_tnn_y = l_model.bias(l_tnn_y);
// 		l_tnn_y = l_model.leaky_relu(l_tnn_y, 0.3);
// 		l_tnn_y = l_model.weight_junction(l_tnn_y, 1);
// 		l_tnn_y = l_model.bias(l_tnn_y);
// 		l_tnn_y = l_model.leaky_relu(l_tnn_y, 0.3);
// 		l_y.push_back(l_tnn_y);
// 	}

// 	auto l_desired_y = input(l_y.size(), l_y[0].size());
// 	auto l_error = l_model.mean_squared_error(l_y, pointers(l_desired_y))->depend();

// 	state_cuboid l_training_set_xs =
// 	{
// 		{
// 			{0, 0},
// 			{0, 1},
// 			{1, 0},
// 			{1, 1}
// 		},
// 		{
// 			{0, 1},
// 			{0, 1},
// 			{1, 0},
// 			{1, 1}
// 		},
// 	};

// 	state_cuboid l_training_set_ys =
// 	{
// 		{
// 			{0},
// 			{1},
// 			{1},
// 			{0}
// 		},
// 		{
// 			{0},
// 			{1},
// 			{1},
// 			{1}
// 		},
// 	};

//     gradient_descent_with_momentum l_optimizer(l_model.parameters(), true, 0.02, 0.9);

// 	size_t CHECKPOINT = 100;

// 	for (int epoch = 0; epoch < 1000000; epoch++)
// 	{
// 		double l_cost = 0;

// 		for (int i = 0; i < l_training_set_xs.size(); i++)
// 		{
// 			set_state(pointers(l_x), l_training_set_xs[i]);
// 			set_state(pointers(l_desired_y), l_training_set_ys[i]);

// 			l_model.fwd();
// 			l_cost += l_error.m_state;
// 			l_error.m_partial_gradient = 1;
// 			l_model.bwd();
// 		}

//         l_optimizer.update();

// 		if (epoch % CHECKPOINT == 0)
// 			std::cout << l_cost << std::endl;

// 	}

// }

// void matrix_vector_multiply_test(

// )
// {
//     model l_model;

// 	sgp_matrix l_x_0
// 	{
// 		{1, 2},
// 		{3, 4},
// 		{5, 6},
// 		{7, 8}
// 	};

// 	sgp_vector l_x_1
// 	{
// 		2,
// 		3
// 	};

// 	auto l_y = l_model.multiply(pointers(l_x_0), pointers(l_x_1));

// 	l_model.fwd();

// }

// void cosine_similarity_test(

// )
// {
//     model l_model;

// 	sgp_vector l_x_0{ 0, 1, 0, 0 };
// 	sgp_vector l_x_1{ 0, -1, 0, 0 };

// 	auto l_y = l_model.cosine_similarity(pointers(l_x_0), pointers(l_x_1));

// 	l_model.fwd();

// }

// void similarity_interpolate_test(

// )
// {
// 	sgp_matrix l_tsx =
// 	{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1},
// 		{1, 0.75}
// 	};

// 	sgp_matrix l_tsy =
// 	{
// 		{0},
// 		{1},
// 		{1},
// 		{0},
// 		{0.862}
// 	};

// 	auto l_query = input(2);
	
//     model l_model;

// 	auto l_y = l_model.similarity_interpolate(pointers(l_query), pointers(l_tsx), pointers(l_tsy));

// 	while (true)
// 	{
// 		std::cout << "INPUT TWO VALUES." << std::endl;
// 		std::cin >> l_query[0].m_state;
// 		std::cin >> l_query[1].m_state;
// 		l_model.fwd();
// 		std::cout << l_y[0]->m_state << std::endl;
// 	}


// }

void large_memory_usage_test(

)
{
    std::cout << "Testing large model" << std::endl;

    constexpr size_t WEIGHT_MATRIX_HEIGHT = 500;

    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

    model::begin();

    std::chrono::time_point l_start = std::chrono::high_resolution_clock::now();
    
	auto l_x = input<WEIGHT_MATRIX_HEIGHT>();

    auto l_w0 = input<WEIGHT_MATRIX_HEIGHT, WEIGHT_MATRIX_HEIGHT>(l_randomly_generate_parameter);
    auto l_w1 = input<WEIGHT_MATRIX_HEIGHT, WEIGHT_MATRIX_HEIGHT>(l_randomly_generate_parameter);
    auto l_w2 = input<WEIGHT_MATRIX_HEIGHT, WEIGHT_MATRIX_HEIGHT>(l_randomly_generate_parameter);

    std::cout << "PARAMETERS CREATED: " << WEIGHT_MATRIX_HEIGHT * WEIGHT_MATRIX_HEIGHT * 3 << " PARAMETERS; " << duration_ms(l_start) << " ms" << std::endl;
    l_start = std::chrono::high_resolution_clock::now();

	{
        auto l_w0_y = multiply(l_w0, l_x);
        auto l_w1_y = multiply(l_w1, l_w0_y);
        auto l_w2_y = multiply(l_w2, l_w1_y);

        model l_model = model::end();

		std::cout << "MODEL CREATED: " << l_model.elements().size() << " elements; " << duration_ms(l_start) << " ms" << std::endl;
        l_start = std::chrono::high_resolution_clock::now();

		l_model.fwd();
		std::cout << "FORWARD COMPLETED: " << duration_ms(l_start) << " ms" << std::endl;
        l_start = std::chrono::high_resolution_clock::now();

		l_model.bwd();
		std::cout << "BACKWARD COMPLETED: " << duration_ms(l_start) << " ms" << std::endl;
        l_start = std::chrono::high_resolution_clock::now();

	}

	std::cout << "DECONSTRUCTED: " << duration_ms(l_start) << " ms" << std::endl;

}

// void pablo_tnn_example(

// )
// {
//     latent::model l_model;

// 	// Write model building code here
// 	sgp_vector l_x = { 0, 0 };

// 	sgp_ptr_vector l_y = pointers(l_x);

// 	l_y = l_model.weight_junction(l_y, 5);
// 	l_y = l_model.bias(l_y);
// 	l_y = l_model.tanh(l_y);
	
// 	l_y = l_model.weight_junction(l_y, 1);
// 	l_y = l_model.bias(l_y);
// 	l_y = l_model.sigmoid(l_y);


// 	auto l_desired_y = input(l_y.size());
// 	auto l_error = l_model.mean_squared_error(l_y, pointers(l_desired_y))->depend();

// 	gradient_descent l_optimizer(l_model.parameters(), true, 0.02);

// 	state_matrix l_tsx =
// 	{
// 		{ 0, 0 },
// 		{ 0, 1 },
// 		{ 1, 0 },
// 		{ 1, 1 },
// 	};

// 	state_matrix l_tsy =
// 	{
// 		{ 0 },
// 		{ 1 },
// 		{ 1 },
// 		{ 0 },
// 	};

// 	const size_t CHECKPOINT = 100000;

// 	for (int epoch = 0; epoch < 1000000; epoch++)
// 	{
// 		for (int i = 0; i < l_tsx.size(); i++)
// 		{
// 			set_state(pointers(l_x), l_tsx[i]);
// 			set_state(pointers(l_desired_y), l_tsy[i]);

// 			l_model.fwd();
// 			l_error.m_partial_gradient = 1;
// 			l_model.bwd();

// 			if (epoch % CHECKPOINT == 0)
// 			{
// 				std::cout << l_y[0]->m_state << std::endl;
// 			}

// 		}

// 		l_optimizer.update();

// 		if (epoch % CHECKPOINT == 0)
// 		{
// 			std::cout << std::endl;
// 		}

// 	}

// }

// void reward_structure_modeling(

// )
// {
// 	sgp_vector l_x(3);

// 	// INPUTS:
// 	// PERCEIVED CHEAPNESS OF THE ITEM
// 	// PREDICTED INCREASE IN UTILITY
// 	// PREDICTED INCREASE IN ENJOYMENT

// 	// OUTPUTS:
// 	// DESIRE TO PURCHASE

//     model l_model;

// 	auto l_normalized_parameters = l_model.normalize(l_model.sigmoid(l_model.parameters(l_x.size())));

// 	auto l_y = l_model.multiply(l_normalized_parameters, pointers(l_x));

// 	auto l_desired_y = state_gradient_pair();
// 	auto l_error = l_model.mean_squared_error(l_y, &l_desired_y)->depend();

// 	gradient_descent l_optimizer(l_model.parameters(), true, 0.02);

// 	struct training_set
// 	{
// 		state_vector m_x;
// 		double m_y;
// 	};

// 	std::vector<training_set> l_training_sets
// 	{
// 		training_set
// 		{
// 			{ -1, 0.1, 0.5 },
// 			0.25
// 		},
// 		training_set
// 		{
// 			{ 0.7, 0.05, 0 },
// 			0.3
// 		},
// 		training_set
// 		{
// 			{ 0.6, 0.05, -0.1 },
// 			-0.3
// 		},

// 	};

// 	for (int epoch = 0; epoch < 1000000; epoch++)
// 	{
// 		double l_cost = 0;

// 		for (auto& l_training_set : l_training_sets)
// 		{
// 			set_state(pointers(l_x), l_training_set.m_x);
// 			l_desired_y.m_state = l_training_set.m_y;

// 			l_model.fwd();

// 			l_cost += l_error.m_state;
// 			l_error.m_partial_gradient = 1;

// 			l_model.bwd();

// 		}

// 		l_optimizer.update();

// 		if (epoch % 10000 == 0)
// 			std::cout << l_cost << std::endl;

// 	}

// 	std::cout << std::endl;
	
// 	for (auto& l_parameter : l_normalized_parameters)
// 		std::cout << l_parameter->m_state << std::endl;

// }

// void loss_modeling_test_0(

// )
// {
// 	sgp_vector l_task_x(10);
// 	sgp_vector l_task_prediction(1);
// 	sgp_vector l_loss_model_desired_y(1);

//     model l_model;

// 	std::vector<size_t> l_tnn_layer_sizes = { 20, 20 };

// 	auto l_loss_model_y = concat(pointers(l_task_x), pointers(l_task_prediction));

// 	for (int i = 0; i < l_tnn_layer_sizes.size(); i++)
// 	{
// 		l_loss_model_y = l_model.weight_junction(l_loss_model_y, l_tnn_layer_sizes[i]);
// 		l_loss_model_y = l_model.bias(l_loss_model_y);
// 		l_loss_model_y = l_model.leaky_relu(l_loss_model_y, 0.3);
// 	}

// 	l_loss_model_y = l_model.weight_junction(l_loss_model_y, 1);
// 	l_loss_model_y = l_model.bias(l_loss_model_y);
// 	l_loss_model_y = l_model.leaky_relu(l_loss_model_y, 0.3);
// 	l_loss_model_y = { l_model.pow(l_loss_model_y[0], l_model.constant(2)) };

// 	auto l_loss_model_loss = l_model.mean_squared_error(l_loss_model_y, pointers(l_loss_model_desired_y))->depend();

// 	gradient_descent l_optimizer(l_model.parameters(), true, 0.02);

// 	std::uniform_real_distribution<double> l_urd(-10, 10);
// 	std::default_random_engine l_dre(28);

// 	for (int epoch = 0; epoch < 1000000; epoch++)
// 	{
// 		double l_loss_model_epoch_loss = 0;

// 		for (int i = 0; i < 1; i++)
// 		{
// 			double l_task_desired_y = 0;

// 			for (int j = 0; j < l_task_x.size(); j++)
// 			{
// 				// GENERATE RANDOM TASK INPUT AND APPLY THE ACTUAL TASK TO THE INPUT (ADDITIVE ACCUMULATION IN THIS CASE)
// 				l_task_x[j].m_state = l_urd(l_dre);
// 				l_task_desired_y += l_task_x[j].m_state;
// 			}

// 			// GENERATE A RANDOM TASK PREDICTION GIVEN THIS INPUT (THIS WILL MOST LIKELY BE WRONG, BUT TO VARYING DEGREES)
// 			l_task_prediction[0].m_state = l_urd(l_dre);

// 			// CALCULATE MEAN SQUARED ERROR OF THE TASK PREDICTION
// 			l_loss_model_desired_y[0].m_state = std::pow(l_task_prediction[0].m_state - l_task_desired_y, 2);

// 			l_model.fwd();
			
// 			l_loss_model_loss.m_partial_gradient = 1;
// 			l_loss_model_epoch_loss += l_loss_model_loss.m_state;

// 			l_model.bwd();

// 		}

// 		l_optimizer.update();

// 		if (epoch % 10000 == 0)
// 		{
// 			std::cout << "LR: " << l_optimizer.m_learn_rate << ", LOSS: " << l_loss_model_epoch_loss << std::endl;
// 			l_optimizer.m_learn_rate *= 0.99;
// 		}

// 	}

// 	std::cout << std::endl << std::endl << "GENERATING TASK PREDICTION: " << std::endl;

// 	gradient_descent l_task_prediction_optimizer(pointers(l_task_prediction), true, 0.2);
	 
// 	double l_task_desired_y = 0;

// 	for (int j = 0; j < l_task_x.size(); j++)
// 	{
// 		// GENERATE RANDOM TASK INPUT AND APPLY THE ACTUAL TASK TO THE INPUT (ADDITIVE ACCUMULATION IN THIS CASE)
// 		l_task_x[j].m_state = l_urd(l_dre);
// 		l_task_desired_y += l_task_x[j].m_state;
// 	}

// 	l_loss_model_desired_y[0].m_state = 0;

// 	for (int epoch = 0; epoch < 100000; epoch++)
// 	{
// 		l_model.fwd();
// 		l_loss_model_loss.m_partial_gradient = 1;
// 		l_model.bwd();
// 		l_task_prediction_optimizer.update();
// 		if (epoch % 10000 == 0)
// 		{
// 			std::cout << 
// 				"PREDICTED TASK Y: " <<
// 				l_task_prediction[0].m_state <<
// 				", DESIRED TASK Y: " <<
// 				l_task_desired_y <<
// 				std::endl;
// 		}
// 	}

// 	std::cout << std::endl << std::endl << "GENERATING TASK X: " << std::endl;

// 	gradient_descent l_task_x_optimizer(pointers(l_task_x), true, 0.2);

// 	l_task_prediction[0].m_state = 10000;
// 	l_loss_model_desired_y[0].m_state = 0;

// 	for (int epoch = 0; epoch < 100000; epoch++)
// 	{
// 		l_model.fwd();
// 		l_loss_model_loss.m_partial_gradient = 1;
// 		l_model.bwd();
// 		l_task_x_optimizer.update();
// 		if (epoch % 10000 == 0)
// 		{
// 			std::cout <<
// 				"TRYING TO ACHIEVE TASK PREDICTION OF: " <<
// 				l_task_prediction[0].m_state <<
// 				", TASK X: ";
			
// 			for (auto& l_value : l_task_x)
// 				std::cout << l_value.m_state << " ";

// 			double l_x_sum = 0;

// 			for (auto& l_value : l_task_x)
// 				l_x_sum += l_value.m_state;

// 			std::cout << "WHICH YIELDS A SUM OF: " << l_x_sum << std::endl;

// 		}
// 	}


// }

// void tnn_test_2(

// )
// {
// 	state_matrix l_tsx =
// 	{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1},
// 	};
// 	state_matrix l_tsy =
// 	{
// 		{0},
// 		{1},
// 		{1},
// 		{0},
// 	};
	
//     model l_model;

// 	auto l_x = input(2);
// 	auto l_y = pointers(l_x);

// 	std::vector<size_t> l_layer_sizes = { 5, 1 };

// 	for (int i = 0; i < l_layer_sizes.size(); i++)
// 	{
// 		l_y = l_model.weight_junction(l_y, l_layer_sizes[i]);
// 		l_y = l_model.bias(l_y);
// 		l_y = l_model.leaky_relu(l_y, 0.3);
// 	}

// 	auto l_desired = input(1);
// 	auto l_loss = l_model.mean_squared_error(l_y, pointers(l_desired))->depend();

// 	gradient_descent l_optimizer(l_model.parameters(), true, 0.02);

// 	const int CHECKPOINT = 100000;

// 	for (int epoch = 0; true; epoch++)
// 	{
// 		for (int i = 0; i < l_tsx.size(); i++)
// 		{
// 			set_state(pointers(l_x), l_tsx[i]);
// 			set_state(pointers(l_desired), l_tsy[i]);
// 			l_model.fwd();
// 			l_loss.m_partial_gradient = 1;
// 			l_model.bwd();
// 			if (epoch % CHECKPOINT == 0)
// 			{
// 				std::cout << l_y[0]->m_state << std::endl;
// 			}
// 		}
// 		l_optimizer.update();
// 		if (epoch % CHECKPOINT == 0)
// 		{
// 			std::cout << std::endl;
// 		}
// 	}

// }

// void convolve_test(

// )
// {
//     model l_model;

// 	auto l_x = input(3, 50, 50);
// 	auto l_filter = input(3, 5, 5);
// 	auto l_y = l_model.convolve(pointers(l_x), pointers(l_filter), 10);

// 	for (int i = 0; i < l_x.size(); i++)
// 	{
// 		for (int j = 0; j < l_x[0].size(); j++)
// 		{
// 			for (int k = 0; k < l_x[0][0].size(); k++)
// 			{
// 				l_x[i][j][k] = (i + j + k) % 100;
// 			}
// 		}
// 	}

// 	for (int i = 0; i < l_filter.size(); i++)
// 	{
// 		for (int j = 0; j < l_filter[0].size(); j++)
// 		{
// 			for (int k = 0; k < l_filter[0][0].size(); k++)
// 			{
// 				l_filter[i][j][k] = (i + j + k) % 100;
// 			}
// 		}
// 	}

// 	l_model.fwd();

// }

// void cnn_test(

// )
// {
//     model l_model;

// 	auto l_x = input(3, 1080, 1920);
// 	auto l_cnn_y = l_model.convolve(pointers(l_x), l_model.parameters(3, 100, 100), 100);
// 	l_cnn_y = l_model.average_pool(l_cnn_y, 3, 3, 3);
// 	l_cnn_y = l_model.leaky_relu(l_cnn_y, 0.3);
// 	l_cnn_y = l_model.convolve({ l_cnn_y }, l_model.parameters(1, 2, 2));

// 	std::vector<size_t> l_layer_sizes = { 15, 2 };

// 	auto l_tnn_y = flatten(l_cnn_y);
	
// 	for (auto& l_layer_size : l_layer_sizes)
// 	{
// 		l_tnn_y = l_model.weight_junction(l_tnn_y, l_layer_size);
// 		l_tnn_y = l_model.bias(l_tnn_y);
// 		l_tnn_y = l_model.leaky_relu(l_tnn_y, 0.3);
// 	}

// }

// void oneshot_matrix_multiply_test(

// )
// {
// 	state_matrix l_matrix = 
// 	{
// 		{2, 3, 4},
// 		{5, 6, 7}
// 	};

// 	state_vector l_weights = { 3, 4, 5 };

// 	state_vector l_result = oneshot::multiply(l_matrix, l_weights);

// }

// void oneshot_tnn_test(

// )
// {
// 	state_matrix l_x = 
// 	{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1}
// 	};

// 	state_matrix l_y =
// 	{
// 		{0},
// 		{1},
// 		{1},
// 		{0}
// 	};

// 	oneshot::parameter_vector_builder l_parameter_vector_builder(-1, 1);
// 	oneshot::parameter_vector& l_parameter_vector(l_parameter_vector_builder);

// 	auto l_carry_forward = 
// 		[&l_parameter_vector, &l_x]
// 	{
// 		state_matrix l_result = l_x;

// 		for (int i = 0; i < l_x.size(); i++)
// 		{
// 			l_parameter_vector.next_index(0);
// 			l_result[i] = oneshot::multiply(l_parameter_vector.next(200, l_result[i].size()), l_result[i]);
// 			l_result[i] = oneshot::add(l_result[i], l_parameter_vector.next(200));
// 			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
// 			l_result[i] = oneshot::multiply(l_parameter_vector.next(1, l_result[i].size()), l_result[i]);
// 			l_result[i] = oneshot::add(l_result[i], l_parameter_vector.next(1));
// 			l_result[i] = oneshot::sigmoid(l_result[i]);
// 		}

// 		return l_result;

// 	};

// 	auto l_dry_fire = l_carry_forward();

// 	state_vector l_gradients(l_parameter_vector.size());

// 	std::uniform_real_distribution<double> l_urd(-1, 1);

// 	for (int i = 0; i < l_gradients.size(); i++)
// 		l_gradients[i] = l_urd(i_default_random_engine);

// 	double l_previous_reward = 1.0;

// 	double l_beta = 0;
// 	double l_learn_rate = 0.0002;

// 	std::uniform_real_distribution<double> l_rcv(-0.001, 0.001);

// 	double l_performance_momentum = 1.0;

// 	for (int epoch = 0; true; epoch++)
// 	{
// 		state_vector l_updates = oneshot::normalize(l_gradients);

// 		for (int i = 0; i < l_gradients.size(); i++)
// 		{
// 			l_updates[i] *= l_learn_rate;
// 			l_updates[i] += l_learn_rate / (std::abs(l_performance_momentum) + l_learn_rate) * l_rcv(i_default_random_engine);
// 			l_parameter_vector[i] += l_updates[i];
// 		}

// 		double l_current_reward = 1.0 / oneshot::mean_squared_error(l_carry_forward(), l_y);
// 		double l_change_in_reward = l_current_reward - l_previous_reward;

// 		l_performance_momentum =
// 			l_beta * l_performance_momentum +
// 			(1.0 - l_beta) * l_change_in_reward;

// 		l_previous_reward = l_current_reward;
		
// 		for (int i = 0; i < l_gradients.size(); i++)
// 		{
// 			double l_instantaneous_gradient_approximation = l_change_in_reward * l_updates[i] / std::pow(oneshot::magnitude(l_updates), 2.0);
// 			l_gradients[i] = l_beta * l_gradients[i] + (1.0 - l_beta) * l_instantaneous_gradient_approximation;
// 		}

// 		if (epoch % 100 == 0)
// 			std::cout << l_current_reward << std::endl;

// 	}

// }

// //void oneshot_tnn_acceleration_test(
// //
// //)
// //{
// //	state_matrix l_x =
// //	{
// //		{0, 0},
// //		{0, 1},
// //		{1, 0},
// //		{1, 1}
// //	};
// //
// //	state_matrix l_y =
// //	{
// //		{0},
// //		{1},
// //		{1},
// //		{0}
// //	};
// //
// //	oneshot::parameter_vector l_position(-1, 1);
// //
// //	auto l_carry_forward =
// //		[&l_position, &l_x]
// //	{
// //		state_matrix l_result = l_x;
// //
// //		for (int i = 0; i < l_x.size(); i++)
// //		{
// //			l_position.next_index(0);
// //			l_result[i] = oneshot::multiply(l_position.next(100, l_result[i].size()), l_result[i]);
// //			l_result[i] = oneshot::add(l_result[i], l_position.next(100));
// //			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
// //			l_result[i] = oneshot::multiply(l_position.next(1, l_result[i].size()), l_result[i]);
// //			l_result[i] = oneshot::add(l_result[i], l_position.next(1));
// //			l_result[i] = oneshot::sigmoid(l_result[i]);
// //		}
// //
// //		return l_result;
// //
// //	};
// //
// //	auto l_dry_fire = l_carry_forward();
// //
// //	std::uniform_real_distribution<double> l_velocity_urd(-0.00001, 0.00001);
// //	state_vector l_velocity(l_position.size());
// //	for (int i = 0; i < l_velocity.size(); i++)
// //		l_velocity[i] = l_velocity_urd(i_default_random_engine);
// //
// //	std::uniform_real_distribution<double> l_acceleration_urd(-0.0001, 0.0001);
// //	state_vector l_acceleration(l_position.size());
// //	for (int i = 0; i < l_acceleration.size(); i++)
// //		l_acceleration[i] = l_acceleration_urd(i_default_random_engine);
// //
// //	std::uniform_real_distribution<double> l_random_velocity_change(-0.001, 0.001);
// //
// //	double l_beta = 0.99;
// //	double l_alpha = 0.002;
// //
// //	double l_previous_reward = 0;
// //	double l_previous_change_in_reward = 0;
// //
// //	for (int l_epoch = 0; true; l_epoch++)
// //	{
// //		state_vector l_velocity_update = oneshot::multiply(oneshot::normalize(l_acceleration), l_alpha);
// //		/*for (int i = 0; i < l_velocity_update.size(); i++)
// //			l_velocity_update[i] += l_alpha * l_random_velocity_change(i_default_random_engine);*/
// //		l_velocity = oneshot::add(l_velocity, l_velocity_update);
// //		l_position = oneshot::add(l_position, l_velocity);
// //		double l_reward = 1.0 / oneshot::mean_squared_error(l_carry_forward(), l_y);
// //		double l_change_in_reward = l_reward - l_previous_reward;
// //		double l_change_in_change_in_reward = l_change_in_reward - l_previous_change_in_reward;
// //		l_previous_reward = l_reward;
// //		l_previous_change_in_reward = l_change_in_reward;
// //		double l_velocity_update_magnitude_squared = std::pow(oneshot::magnitude(l_velocity_update), 2);
// //		for (int i = 0; i < l_acceleration.size(); i++)
// //		{
// //			double l_instant_acceleration = l_change_in_change_in_reward * l_velocity_update[i] / l_velocity_update_magnitude_squared;
// //			// Construct a running average for the acceleration
// //			l_acceleration[i] = l_beta * l_acceleration[i] + (1 - l_beta) * l_instant_acceleration;
// //		}
// //		std::cout << l_reward << std::endl;
// //		Sleep(100);
// //	}
// //
// //
// //}

// void particle_swarm_optimization_example(

// )
// {
// 	state_matrix l_x =
// 	{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1}
// 	};

// 	state_matrix l_y =
// 	{
// 		{0},
// 		{1},
// 		{1},
// 		{0}
// 	};

// 	auto l_carry_forward =
// 		[&l_x](oneshot::parameter_vector& a_parmeter_vector)
// 	{
// 		state_matrix l_result = l_x;

// 		for (int i = 0; i < l_x.size(); i++)
// 		{
// 			a_parmeter_vector.next_index(0);
// 			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1000, l_result[i].size()), l_result[i]);
// 			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1000));
// 			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
// 			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1, l_result[i].size()), l_result[i]);
// 			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1));
// 			l_result[i] = oneshot::sigmoid(l_result[i]);
// 		}

// 		return l_result;

// 	};

// 	// Initialize particle positions
// 	std::vector<oneshot::parameter_vector> l_particle_positions;
// 	for (int i = 0; i < 100; i++)
// 	{
// 		oneshot::parameter_vector_builder l_builder(-1, 1);
// 		l_carry_forward(l_builder); // Dry fire the particle's parameter vector
// 		l_particle_positions.push_back(l_builder);
// 	}

// 	// Initialize particle velocities
// 	state_matrix l_particle_velocities = oneshot::make(
// 		l_particle_positions.size(),
// 		l_particle_positions[0].size()
// 	);

// 	// Define hyperparameters
// 	double l_w = 0.9;
// 	double l_c1 = 0.4;
// 	double l_c2 = 0.6;

// 	state_matrix l_p_best = oneshot::make(l_particle_positions.size(), l_particle_positions[0].size());
// 	state_vector l_p_best_losses(l_particle_positions.size());
// 	for (int i = 0; i < l_p_best_losses.size(); i++)
// 		l_p_best_losses[i] = 9999999999999999;

// 	state_vector l_g_best(l_particle_positions[0].size());
// 	double l_g_best_loss = 9999999999999999;

// 	// Train
// 	for (int epoch = 0; true; epoch++)
// 	{
// 		// Evaluate the loss at each particle's position
// 		for (int i = 0; i < l_particle_positions.size(); i++)
// 		{
// 			double l_loss = oneshot::mean_squared_error(l_carry_forward(l_particle_positions[i]), l_y);
// 			if (l_loss < l_p_best_losses[i])
// 			{
// 				l_p_best[i] = l_particle_positions[i];
// 				l_p_best_losses[i] = l_loss;
// 			}
// 			if (l_loss < l_g_best_loss)
// 			{
// 				l_g_best = l_particle_positions[i];
// 				l_g_best_loss = l_loss;
// 			}
// 		}

// 		// Update the velocities of all particles
// 		for (int i = 0; i < l_particle_positions.size(); i++)
// 		{
// 			state_vector l_weighted_particle_velocity = oneshot::multiply(l_particle_velocities[i], l_w);
// 			state_vector l_cognitive_term = oneshot::multiply(oneshot::multiply(oneshot::subtract(l_p_best[i], l_particle_positions[i]), l_c1), oneshot::random(0, 1));
// 			state_vector l_social_term = oneshot::multiply(oneshot::multiply(oneshot::subtract(l_g_best, l_particle_positions[i]), l_c2), oneshot::random(0, 1));
// 			l_particle_velocities[i] = oneshot::add(oneshot::add(l_weighted_particle_velocity, l_cognitive_term), l_social_term);
// 		}

// 		// Update the positions of all particles
// 		for (int i = 0; i < l_particle_positions.size(); i++)
// 		{
// 			l_particle_positions[i] = oneshot::add(l_particle_positions[i], l_particle_velocities[i]);
// 		}

// 		std::cout << l_g_best_loss << std::endl;

// 	}

// }

// void particle_swarm_optimization_class_example(

// )
// {
// 	state_matrix l_x =
// 	{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1}
// 	};

// 	state_matrix l_y =
// 	{
// 		{0},
// 		{1},
// 		{1},
// 		{0}
// 	};

// 	auto l_carry_forward =
// 		[&l_x](oneshot::parameter_vector& a_parmeter_vector)
// 	{
// 		state_matrix l_result = l_x;

// 		for (int i = 0; i < l_x.size(); i++)
// 		{
// 			a_parmeter_vector.next_index(0);
// 			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1000, l_result[i].size()), l_result[i]);
// 			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1000));
// 			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
// 			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1, l_result[i].size()), l_result[i]);
// 			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1));
// 			l_result[i] = oneshot::sigmoid(l_result[i]);
// 		}

// 		return l_result;

// 	};

// 	// Initialize particle positions
// 	std::vector<oneshot::parameter_vector> l_particle_positions;

// 	for (int i = 0; i < 100; i++)
// 	{
// 		oneshot::parameter_vector_builder l_builder(-1, 1);
// 		l_carry_forward(l_builder); // Dry fire the particle's parameter vector
// 		l_particle_positions.push_back(l_builder);
// 	}

// 	// Define hyperparameters
// 	double l_w = 0.9;
// 	double l_c1 = 0.2;
// 	double l_c2 = 0.8;

// 	std::vector<oneshot::particle_optimizer> l_particles;
// 	for (int i = 0; i < l_particle_positions.size(); i++)
// 		l_particles.push_back(oneshot::particle_optimizer(l_particle_positions[i]));

// 	oneshot::particle_swarm_optimizer l_swarm_optimizer(l_particles, l_w, l_c1, l_c2);

// 	state_vector l_particle_rewards(l_particles.size());

// 	// Train
// 	for (int epoch = 0; true; epoch++)
// 	{
// 		for (int i = 0; i < l_particles.size(); i++)
// 			l_particle_rewards[i] = 1.0 / (oneshot::mean_squared_error(l_carry_forward(l_particle_positions[i]), l_y) + 0.00001);
// 		l_swarm_optimizer.update(l_particle_rewards);
// 		std::cout << l_swarm_optimizer.global_best_reward() << std::endl;
// 	}

// }

// void oneshot_partition_test(

// )
// {
// 	auto l_tensor = oneshot::random(10, 10, 10, 0, 1);
// 	auto l_flattened_0 = oneshot::flatten(l_tensor);
// 	auto l_tensor_recovered = oneshot::partition(l_flattened_0, l_tensor.size(), l_tensor[0].size(), l_tensor[0][0].size());

// 	assert(l_tensor_recovered == l_tensor);

// 	auto l_matrix = oneshot::random(10, 10, 0, 1);
// 	auto l_flattened_1 = oneshot::flatten(l_matrix);
// 	auto l_matrix_recovered = oneshot::partition(l_flattened_1, l_tensor.size(), l_tensor[0].size());

// 	assert(l_matrix_recovered == l_matrix);

// }

// void scalar_multiplication_modeling_using_matrices(

// )
// {
//     model l_model;

// 	auto l_x = input(4);

// 	auto l_m = l_model.bias(l_model.weight_junction(pointers(l_x), 20));
// 	l_m = l_model.bias(l_model.weight_junction(l_m, 20));
// 	auto l_y = l_model.bias(l_model.weight_junction(l_m, 1));

// 	auto l_desired = input(1);
// 	auto l_loss = l_model.mean_squared_error(l_y, pointers(l_desired))->depend();

// 	gradient_descent_with_momentum l_optimizer(l_model.parameters(), true, 0.2, 0.9);

// 	state_matrix l_ts_x =
// 	{
// 		{0, 0},
// 		{0, 1},
// 		{1, 0},
// 		{1, 1},
// 	};

// 	state_matrix l_ts_y =
// 	{
// 		{0},
// 		{1},
// 		{1},
// 		{1}
// 	};

// 	for (int epoch = 0; true; epoch++)
// 	{
// 		double l_epoch_cost = 0;
// 		for (int i = 0; i < l_ts_x.size(); i++)
// 		{
// 			set_state(pointers(l_x), l_ts_x[i]);
// 			set_state(pointers(l_desired), l_ts_y[i]);
// 			l_model.fwd();
// 			l_loss.m_partial_gradient = 1;
// 			l_model.bwd();
// 			l_epoch_cost += l_loss.m_state;
// 		}
// 		l_optimizer.update();

// 		if (l_epoch_cost <= 0.3)
// 			printf("123");

// 		if (epoch % 10000 == 0)
// 			std::cout << l_epoch_cost << std::endl;
// 	}

// }

void test_pso(

)
{
    std::cout << "TESTING PARTICLE SWARM OPTIMIZATION" << std::endl;

    constexpr size_t X_HEIGHT = 4;
    constexpr size_t X_WIDTH =  2;
    constexpr size_t Y_HEIGHT = 4;
    constexpr size_t Y_WIDTH =  1;
    constexpr size_t PARTICLE_COUNT = 20;

    constexpr std::array<size_t, 3> TNN_DIMS = {2, 5, 1};

    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };


    std::stringstream l_tsx_ss(
        "0 0\n"
        "0 1\n"
        "1 0\n"
        "1 1\n"
    );
    
    std::stringstream l_tsy_ss(
        "0\n"
        "1\n"
        "1\n"
        "0\n"
    );
    
    tensor<double, X_HEIGHT, X_WIDTH> l_tsx;
    tensor<double, Y_HEIGHT, Y_WIDTH> l_tsy;

    l_tsx_ss >> l_tsx;
    l_tsy_ss >> l_tsy;

    auto l_b1 = constant<double, TNN_DIMS[1]>();
    auto l_b2 = constant<double, TNN_DIMS[2]>();
    auto l_w0 = constant<double, TNN_DIMS[1], TNN_DIMS[0]>();
    auto l_w1 = constant<double, TNN_DIMS[2], TNN_DIMS[1]>();
    
    constexpr size_t PARAMETER_VECTOR_SIZE = 
        l_b1.flattened_size() +
        l_b2.flattened_size() +
        l_w0.flattened_size() +
        l_w1.flattened_size();

    auto l_positions = constant<double, PARTICLE_COUNT, PARAMETER_VECTOR_SIZE>(l_randomly_generate_parameter);

    auto l_get_y = [&](
        const auto& a_parameter_vector
    )
    {
        // This copies the parameters into their respective places.
        copy(a_parameter_vector, l_b1, l_b2, l_w0, l_w1);

        auto l_y = constant<double, Y_HEIGHT, Y_WIDTH>();

		for (int i = 0; i < X_HEIGHT; i++)
		{
            auto l_w0_y =   multiply(l_w0, l_tsx[i]);
            auto l_b1_y =   add(l_b1, l_w0_y);
            auto l_tanh_1 = tanh(l_b1_y);
            auto l_w1_y =   multiply(l_w1, l_tanh_1);
            auto l_b2_y =   add(l_b2, l_w1_y);
            auto l_tanh_2 = tanh(l_b2_y);
            l_y[i] = l_tanh_2;
		}

        return l_y;
        
    };

	auto l_get_reward = [&](
		const auto& a_parameter_vector
	)
	{
		return 1.0 / (mean_squared_error(l_get_y(a_parameter_vector), l_tsy) + 1E-10);
	};

	// Initialize the swarm optimizer.
	particle_swarm_optimizer<PARTICLE_COUNT, PARAMETER_VECTOR_SIZE> 
        l_particle_swarm_optimizer(l_positions, 0.9, 0.2, 0.8);

	// Construct a vector of the rewards associated with each parameter vector.
	auto l_rewards = constant<double, PARTICLE_COUNT>();

	for (int l_epoch = 0; l_epoch < 100; l_epoch++)
	{
		for (int i = 0; i < PARTICLE_COUNT; i++)
		{
			l_rewards[i] = l_get_reward(
				l_positions[i]
			);
		}

		l_particle_swarm_optimizer.update(l_rewards);
        
		if (l_epoch % 10 == 0)
			std::cout << l_particle_swarm_optimizer.global_best_reward() << std::endl;
        
	}

    auto l_y_final = l_get_y(l_particle_swarm_optimizer.global_best_position());

    std::cout << std::endl << "DESIRED: " << std::endl << l_tsy << std::endl;
    std::cout << "Actual: " << std::endl <<  l_y_final << std::endl;

    assert(l_y_final[0][0] < 0.1);
    assert(l_y_final[1][0] > 0.9);
    assert(l_y_final[2][0] > 0.9);
    assert(l_y_final[3][0] < 0.1);

}

template<size_t I>
bool nor(
    const tensor<bool, I>& a_connections,
    const tensor<bool, I>& a_x
)
{
    bool l_result = false;

    for (int i = 0; i < a_x.size(); i++)
    {
        if (a_connections[i] && a_x[i])
        {
            l_result = true;
            break;
        }
    }

    return !l_result;
    
};

template<size_t I, size_t J>
tensor<bool, I> nor_layer(
    const tensor<bool, I, J>& a_layer_matrix,
    const tensor<bool, J>& a_x
)
{
    tensor<bool, I> l_result;

    for (int i = 0; i < I; i++)
        l_result[i] = nor(a_layer_matrix[i], a_x);

    return l_result;
    
};

void test_icpso(

)
{
    std::cout << "TESTING INTEGER CATEGORICAL PARTICLE SWARM OPTIMIZATION" << std::endl;

    constexpr size_t X_HEIGHT = 4;
    constexpr size_t X_WIDTH = 2;
    constexpr size_t Y_HEIGHT = 4;
    constexpr size_t Y_WIDTH = 1;
    constexpr size_t PARTICLE_COUNT = 20;
    constexpr std::array<size_t, 5> LAYER_DIMS = {2, 2, 2, 1, 1};

    tensor<bool, LAYER_DIMS[1], LAYER_DIMS[0]> l_layer_1;
    tensor<bool, LAYER_DIMS[2], LAYER_DIMS[0] + LAYER_DIMS[1]> l_layer_2;
    tensor<bool, LAYER_DIMS[3], LAYER_DIMS[1] + LAYER_DIMS[2]> l_layer_3;
    tensor<bool, LAYER_DIMS[4], LAYER_DIMS[2] + LAYER_DIMS[3]> l_layer_4;

    constexpr size_t PARAM_VECTOR_SIZE =
        l_layer_1.flattened_size() +
        l_layer_2.flattened_size() +
        l_layer_3.flattened_size() +
        l_layer_4.flattened_size();
    
    auto l_distribution_sizes = constant<size_t, PARAM_VECTOR_SIZE>(2);

    std::stringstream l_tsx_ss(
        "0 0\n"
        "0 1\n"
        "1 0\n"
        "1 1\n"
    );
    
    std::stringstream l_tsy_ss(
        "0\n"
        "1\n"
        "1\n"
        "0\n"
    );
    
    tensor<bool, X_HEIGHT, X_WIDTH> l_tsx;
    tensor<bool, Y_HEIGHT, Y_WIDTH> l_tsy;

    l_tsx_ss >> l_tsx;
    l_tsy_ss >> l_tsy;

    auto l_get_y = [&](
        const tensor<size_t, PARAM_VECTOR_SIZE>& a_candidate_solution
    )
    {

        tensor<bool, PARAM_VECTOR_SIZE> l_converted_candidate_solution;

        // Convert the global best solution to a valid type (boolean tensor).
        std::transform(a_candidate_solution.begin(), a_candidate_solution.end(), l_converted_candidate_solution.begin(),
            [](const size_t& a_selected_variable_index)
            {
                return a_selected_variable_index != 0;
            });
        
        // This copies the parameters into their respective places.
        copy(l_converted_candidate_solution,
            l_layer_1,
            l_layer_2,
            l_layer_3,
            l_layer_4);

        tensor<bool, Y_HEIGHT, Y_WIDTH> l_y;

		for (int i = 0; i < X_HEIGHT; i++)
		{
            auto l_layer_1_y = nor_layer(l_layer_1, l_tsx[i]);
            auto l_layer_2_y = nor_layer(l_layer_2, concat(l_tsx[i], l_layer_1_y));
            auto l_layer_3_y = nor_layer(l_layer_3, concat(l_layer_1_y, l_layer_2_y));
            auto l_layer_4_y = nor_layer(l_layer_4, concat(l_layer_2_y, l_layer_3_y));
            l_y[i] = l_layer_4_y;
		}

        return l_y;
        
    };

	auto l_get_reward = [&](
		const auto& a_candidate_solution
	)
	{
        double l_result = 0;

        auto l_y = l_get_y(a_candidate_solution);

        for (int i = 0; i < Y_HEIGHT; i++)
            l_result += double(l_y[i] == l_tsy[i]);
        
        return l_result;
        
	};

    icpso<size_t, PARTICLE_COUNT, PARAM_VECTOR_SIZE> l_icpso(
        l_distribution_sizes,
        0.9,
        0.2,
        0.8,
        0.99
    );
    
	// Construct a vector of the rewards associated with each parameter vector.
	auto l_rewards = constant<double, PARTICLE_COUNT>();

	for (int l_epoch = 0; l_epoch < 1000; l_epoch++)
	{
        const tensor<size_t, PARTICLE_COUNT, PARAM_VECTOR_SIZE>& l_candidates = l_icpso.candidate_solutions();
        
		for (int i = 0; i < PARTICLE_COUNT; i++)
		{
			l_rewards[i] = l_get_reward(
				l_candidates[i]
			);
		}

		l_icpso.update(l_rewards);
        
		if (l_epoch % 10 == 0)
			std::cout << l_icpso.global_best_reward() << std::endl;
        
	}

    auto l_y_final = l_get_y(l_icpso.global_best_solution());

    std::cout << std::endl << "DESIRED: " << std::endl << l_tsy << std::endl;
    std::cout << "Actual: " << std::endl <<  l_y_final << std::endl;

    assert(l_y_final[0][0] == false);
    assert(l_y_final[1][0] == true);
    assert(l_y_final[2][0] == true);
    assert(l_y_final[3][0] == false);

}

// void oneshot_convolve_test(

// )
// {
// 	auto l_x = oneshot::random(3, 100, 100, 0, 1);
// 	auto l_filter = oneshot::random(3, 10, 10, 0, 1);
// 	auto l_convolved = oneshot::convolve(l_x, l_filter, 1);
// 	auto l_convolved_first_element =
// 		oneshot::multiply(
// 			oneshot::flatten(oneshot::range(l_x, 0, 0, 0, 3, 10, 10)),
// 			oneshot::flatten(l_filter));
// }

// void sife_concurrent_feature_extraction_0(

// )
// {
//     state_matrix l_ts_x;
//     state_matrix l_ts_y;

//     const size_t INPUT_WIDTH = 10;
//     const size_t FEATURE_VECTOR_WIDTH = 10;
//     const size_t IMAGE_WIDTH = 100;
//     const std::vector<size_t> FEATURE_MODEL_DIMS = { 20, FEATURE_VECTOR_WIDTH };
//     const std::vector<size_t> CHASING_MODEL_DIMS = { 20, 1 };

//     ////////////////////////////////////////////////////////
//     // First, create the f model. (Maps from Q1 to F) //////
//     ////////////////////////////////////////////////////////

//     model l_f;
    
//     auto l_f_x = input(INPUT_WIDTH);
    
//     auto l_f_y = pointers(l_f_x);

//     for (size_t l_layer_size : FEATURE_MODEL_DIMS)
//     {
//         l_f_y = l_f.weight_junction(l_f_y, l_layer_size);
//         l_f_y = l_f.bias(l_f_y);
//         l_f_y = l_f.leaky_relu(l_f_y, 0.3);
//     }


//     ////////////////////////////////////////////////////////
//     // Next, create the g model. (Maps from Q2 to F) //////
//     ////////////////////////////////////////////////////////
    
//     model l_g;

//     auto l_g_x = input(IMAGE_WIDTH);

//     auto l_g_y = pointers(l_g_x);

//     for (size_t l_layer_size : FEATURE_MODEL_DIMS)
//     {
//         l_g_y = l_g.weight_junction(l_g_y, l_layer_size);
//         l_g_y = l_g.bias(l_g_y);
//         l_g_y = l_g.leaky_relu(l_g_y, 0.3);
//     }


//     ////////////////////////////////////////////////////////
//     // Next, define the characteristic loss. ///////////////
//     ////////////////////////////////////////////////////////

//     model l_characteristic_loss;

//     auto l_characteristic_loss_y = l_characteristic_loss.mean_squared_error(l_f_y, l_g_y)->depend();


//     ////////////////////////////////////////////////////////
//     // Next, define the chasing models. ////////////////////
//     ////////////////////////////////////////////////////////

//     model l_h;

//     sgp_ptr_matrix l_h_y(FEATURE_VECTOR_WIDTH);

//     // DEFINE THE INPUT MATRIX TO THE CHASING MODELS
//     for (int i = 0; i < FEATURE_VECTOR_WIDTH; i++)
//     {
//         for (int j = 0; j < FEATURE_VECTOR_WIDTH; j++)
//         {
//             if (i == j)
//                 continue;
            
//             // Average the individual feature values l_f[j] and l_g[j].
//             // Then, we negate the average to cause gradient to be negated in the backward pass.
//             // This is because the independence loss is supplying gradient to
//             // l_f_y and l_g_y, but we actually want l_f_y and l_g_y 
//             // to be supplied dependence loss instead.

//             l_h_y[i].push_back(l_h.negate(l_h.average(std::vector{l_f_y[j], l_g_y[j]})));

//         }
//     }

//     for (auto& l_h_y_row : l_h_y)
//     {
//         for (size_t l_layer_size : CHASING_MODEL_DIMS)
//         {
//             l_h_y_row = l_h.weight_junction(l_h_y_row, l_layer_size);
//             l_h_y_row = l_h.bias(l_h_y_row);
//             l_h_y_row = l_h.leaky_relu(l_h_y_row, 0.3);
//         }
//     }

//     // F0^ G0^
//     // F1^ G1^
//     // F2^ G2^
//     // F3^ G3^

//     auto l_shared_feature_predictions = flatten(l_h_y);

//     // We negate the average of the feature values because
//     // we need the independence loss to become dependence loss.

//     auto l_independence_loss =
//         l_h.mean_squared_error(
//             l_shared_feature_predictions,
//             l_h.negate(l_h.average({l_f_y, l_g_y}))
//         )->depend();
    

//     ////////////////////////////////////////////////////////
//     // Next, we will define the optimizers. ////////////////
//     ////////////////////////////////////////////////////////
//     gradient_descent_with_momentum l_feature_optimizer(concat(l_f.parameters(), l_g.parameters()), true, 0.02, 0.9);
//     gradient_descent_with_momentum l_chasing_optimizer(l_h.parameters(), true, 0.02, 0.9);
    
    
//     ////////////////////////////////////////////////////////
//     // Next, we will define the training loop. /////////////
//     ////////////////////////////////////////////////////////

//     for (int l_epoch = 0; true; l_epoch++)
//     {
        
//     }
    




// }

// void yas(

// )
// {
//     std::uniform_real_distribution<double> l_training_example_distribution(0, 10);
//     std::default_random_engine             l_dre(25);

//     // Construct the neural network. This means constructing a composite function of a bunch of
//     // matrix multiplications and other functions thrown in.

//     auto x = input(1);
//     auto y_hat = pointers(x);

//     model l_model;

//     for (size_t s : {10, 5, 1})
//     {
//         y_hat = l_model.weight_junction(y_hat, s);
//         y_hat = l_model.bias(y_hat);
//         y_hat = l_model.leaky_relu(y_hat, 0.3);
//     }

//     auto y_desired = input(1);

//     auto loss = l_model.mean_squared_error(y_hat, pointers(y_desired))->depend();

//     // Define the optimizer (gradient descent with momentum)
//     gradient_descent optimizer(l_model.parameters(), true, 0.02);



//     const size_t TRAINING_EXAMPLES_PER_EPOCH = 10;


//     // An epoch is like a time period. (like an iteration)
//     for (int epoch = 0; true; epoch++)
//     {
//         double loss_average = 0;

//         double previous_x = 0;
//         double previous_y = 0;
//         double previous_y_hat = 0;

//         for (int training_example_index = 0;
//             training_example_index < TRAINING_EXAMPLES_PER_EPOCH;
//             training_example_index++)
//         {
//             x[0].m_state = l_training_example_distribution(l_dre);// generate random number
//             y_desired[0].m_state = x[0].m_state * x[0].m_state;// calculate square of x
//             l_model.fwd();
//             loss.m_partial_gradient = 1;
//             l_model.bwd();
//             loss_average += loss.m_state;
//             previous_x = x[0].m_state;
//             previous_y = y_desired[0].m_state;
//             previous_y_hat = y_hat[0]->m_state;
//         }

//         loss_average /= (double)TRAINING_EXAMPLES_PER_EPOCH;
        
//         optimizer.update();

//         if (epoch % 100000 == 0)
//             std::cout
//                 << "Average Loss: "
//                 << loss_average
//                 << ", X: "
//                 << previous_x
//                 << ", Y: "
//                 << previous_y
//                 << ", Y_HAT: "
//                 << previous_y_hat
//                 << std::endl;

//     }


// }

/// Linearization of NAND
void nonlinear_scatter_span_linearization(

)
{
    // First, define the constants involved.
    constexpr size_t NODE_COUNT = 5;
    constexpr size_t PARTICLE_COUNT = 25;
    constexpr size_t NUMBER_OF_EVALUATIONS_IN_SPAN = 1000;
    constexpr size_t NUMBER_OF_SPANS = 2;
    
    // In this scatter span, we start off with a list of current states.
    // Then, we repeatedly do the following:
    // Select a random 2-permutation (x0, x1) of the list of current states.
    // Input the 2-perm into the operation N.
    // Replace one of the input operands in the list of current states with the output of N.

    constexpr size_t WAVEFORM_SIZE = 10;
    constexpr std::array<size_t, 3> R_DIMS = { NUMBER_OF_SPANS * WAVEFORM_SIZE, 256, 1 };

    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    std::uniform_real_distribution<double> l_waveform_urd(-10, 10);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };
    std::function<double()> l_randomly_generate_waveform_value = [&l_dre, &l_waveform_urd] { return l_waveform_urd(l_dre); };

    auto l_R_w0 =        constant<double, R_DIMS[1], R_DIMS[0]>();
    auto l_R_b1 =        constant<double, R_DIMS[1]>();
    auto l_R_w1 =        constant<double, R_DIMS[2], R_DIMS[1]>();
    auto l_R_b2 =        constant<double, R_DIMS[2]>();
    auto l_Ns =          constant<double, NUMBER_OF_SPANS, WAVEFORM_SIZE, 2 * WAVEFORM_SIZE>();
    auto l_zero_states = constant<double, NUMBER_OF_SPANS, NODE_COUNT, WAVEFORM_SIZE>();
    auto l_one_states =  constant<double, NUMBER_OF_SPANS, NODE_COUNT, WAVEFORM_SIZE>();

    constexpr size_t PARAMETER_VECTOR_SIZE =
        l_R_w0.flattened_size() +
        l_R_b1.flattened_size() +
        l_R_w1.flattened_size() +
        l_R_b2.flattened_size() +
        l_Ns.flattened_size() +
        l_zero_states.flattened_size() +
        l_one_states.flattened_size();

    auto l_R = [&](
        const tensor<double, NUMBER_OF_SPANS, WAVEFORM_SIZE>& a_x
    )
    {
        auto l_x = flatten(a_x);
        auto l_w0_y = multiply(l_R_w0, l_x);
        auto l_b1_y = add(l_w0_y, l_R_b1);
        auto l_act_1 = leaky_relu(l_b1_y, 0.3);
        auto l_w1_y = multiply(l_R_w1, l_act_1);
        auto l_b2_y = add(l_w1_y, l_R_b2);
        auto l_act_2 = leaky_relu(l_b2_y, 0.3);
        return l_act_2[0] - floor(l_act_2[0]);
    };
    
    auto l_get_scattered_reward = [&](
        auto& a_parameter_vector
    )
    {
        // Populates the parameters into their respective places.
        copy(a_parameter_vector, l_R_w0, l_R_b1, l_R_w1, l_R_b2, l_Ns, l_zero_states, l_one_states);

        // Allocate memory for waveform labels.
        tensor<size_t, NODE_COUNT> l_zero_node_depths;
        tensor<size_t, NODE_COUNT> l_one_node_depths;

        // Initialize node depths to a reasonable value
        for (int i = 0; i < NODE_COUNT; i++)
        {
            l_zero_node_depths[i] = 1;
            l_one_node_depths[i] =  1;
        }

        // THE TRIVIAL SOLUTION IS NOT POSSIBLE HERE, SINCE
        // IF ALL WAVEFORMS ARE CONSIDERED TO BE 0's, THEN 
        // BY EVALUATING N ON THEM ONCE, IT SHOULD RETURN 1.
        // THE CONVERSE IS ALSO TRUE.

        double l_total_accuracy = 0;

        bool l_generate_zero = true;

        // constexpr size_t DEPTH_ACCURACY_HIST_BIN_SIZE = 10;
        // constexpr size_t HIST_BIN_COUNT = NUMBER_OF_EVALUATIONS_IN_SPAN / DEPTH_ACCURACY_HIST_BIN_SIZE;
        
        // auto l_depth_accuracies = constant<double, HIST_BIN_COUNT>(0);
        // auto l_depth_accuracies_bin_counts = constant<double, HIST_BIN_COUNT>(0);

        for (int i = 0; i < NUMBER_OF_EVALUATIONS_IN_SPAN; i++)
        {
            // We will do this a number of times equal to the
            // number of evaluations we should have in each span.
            
            size_t l_random_index_0 = rand() % NODE_COUNT;
            size_t l_random_index_1 = rand() % NODE_COUNT;

            tensor<double, NUMBER_OF_SPANS, NODE_COUNT, WAVEFORM_SIZE>* l_pool_0 = nullptr;
            tensor<double, NUMBER_OF_SPANS, NODE_COUNT, WAVEFORM_SIZE>* l_pool_1 = nullptr;
            tensor<double, NUMBER_OF_SPANS, NODE_COUNT, WAVEFORM_SIZE>* l_replacement_pool = nullptr;

            double l_R_y_label = 0;

            if (l_generate_zero)
            {
                // To generate a zero, since NAND only outputs 0 when both
                // inputs are 1's, we need to sample two waveforms from
                // the 1-state pool.

                l_pool_0 = &l_one_states;
                l_pool_1 = &l_one_states;
                l_replacement_pool = &l_zero_states;

                l_R_y_label = 0.0;

            }
            else
            {
                // There are three ways in which we can produce 1's.
                // (0, 0) -> 1
                // (0, 1) -> 1
                // (1, 0) -> 1
                // We wish to select a random option.

                switch(rand() % 3)
                {
                    case 0:
                        l_pool_0 = &l_zero_states;
                        l_pool_1 = &l_zero_states;
                        break;
                    case 1:
                        l_pool_0 = &l_zero_states;
                        l_pool_1 = &l_one_states;
                        break;
                    case 2:
                        l_pool_0 = &l_one_states;
                        l_pool_1 = &l_zero_states;
                        break;
                    default:
                        break;
                }

                l_replacement_pool = &l_one_states;

                l_R_y_label = 1.0;

            }

            tensor<double, NUMBER_OF_SPANS, WAVEFORM_SIZE> l_N_ys;

            // Compute outputs through the linear operation N.
            for (int i = 0; i < NUMBER_OF_SPANS; i++)
            {
                l_N_ys[i] = multiply(
                    l_Ns[i],
                    concat(
                        (*l_pool_0)[i][l_random_index_0],
                        (*l_pool_1)[i][l_random_index_1]
                    )
                );
            }

            double l_R_y = l_R(l_N_ys);
            
            double l_current_node_accuracy = (1.0 - std::abs(l_R_y - l_R_y_label));

            l_total_accuracy += l_current_node_accuracy;
            
            // Now that we've collected reward, go ahead and replace one of
            // the operands with the output and correct label.

            size_t l_replacement_index = rand() % NODE_COUNT;

            for (int i = 0; i < NUMBER_OF_SPANS; i++)
            {
                (*l_replacement_pool)[i][l_replacement_index] = l_N_ys[i];
            }

            // For the next iteration, we want to do the opposite. (To keep sampling 50-50 zeroes and ones).
            l_generate_zero = !l_generate_zero;

        }

        // for (int i = 0; i < HIST_BIN_COUNT; i++)
        // {
        //     l_depth_accuracies[i] /= l_depth_accuracies_bin_counts[i];
        // }

        l_total_accuracy /= (double)NUMBER_OF_EVALUATIONS_IN_SPAN;
        // l_proportion_of_1_labels /= (double)NUMBER_OF_EVALUATIONS_IN_SPAN;
        // l_proportion_of_1_outputted_bits /= (double)NUMBER_OF_EVALUATIONS_IN_SPAN;

        // if (a_print_accuracy_vector)
        // {
        //     for (int i = 0; i < HIST_BIN_COUNT; i++)
        //     {
        //         if (l_depth_accuracies_bin_counts[i] != 0)
        //             std::cout << l_depth_accuracies[i] << ' ';
        //     }
        //     std::cout << std::endl;
        //     std::cout << "PROPORTION OF 1 LABELS:         " << l_proportion_of_1_labels << std::endl;
        //     std::cout << "PROPORTION OF OUTPUTTED 1 BITS: " << l_proportion_of_1_outputted_bits << std::endl;
        // }

        return l_total_accuracy;
        
    };

    auto l_average_scattered_reward = [&](
        auto& a_parameter_vector,
        const size_t& a_trials
    )
    {
        double l_accuracy_accumulator = 0;
        for (int i = 0; i < a_trials; i++)
            l_accuracy_accumulator += l_get_scattered_reward(a_parameter_vector);
        return l_accuracy_accumulator / (double)a_trials;
    };

    // Randomly generate initial particle positions
    auto l_positions = constant<double, PARTICLE_COUNT, PARAMETER_VECTOR_SIZE>(l_randomly_generate_parameter);

    const std::string PARAM_FILE_NAME = "nonlinear_scatter_span_linearization_params.bin";

    std::ifstream l_param_ifs(PARAM_FILE_NAME, std::ios::binary);

    if (l_param_ifs.is_open())
        l_param_ifs >> l_positions;

    particle_swarm_optimizer l_optimizer(l_positions, 0.9, 0.2, 0.8, 1.0);

    auto l_rewards = constant<double, PARTICLE_COUNT>(-INFINITY);

    double l_previous_best_reward = -INFINITY;

    for (int l_epoch = 0; true; l_epoch++)
    {
        for (int i = 0; i < PARTICLE_COUNT; i++)
            // Get readings of rewards
            l_rewards[i] = l_average_scattered_reward(l_positions[i], 5);

        l_optimizer.update(l_rewards);

        if (l_epoch % 1 == 0)
        {
            std::cout << "AVERAGE: " << average(l_rewards) << ", BEST: ";
            if (l_optimizer.global_best_reward() > l_previous_best_reward)
            {
                std::cout << "(NEW) ";
                l_previous_best_reward = l_optimizer.global_best_reward();
            }
            std::cout << l_optimizer.global_best_reward() << std::endl;

            std::ofstream l_ofs(PARAM_FILE_NAME, std::ios::binary);

            l_ofs << l_positions;

            l_ofs.close();

        }
        
    }

}

bool is_close_to(
    const double& a_x_0,
    const double& a_x_1,
    const double& a_threshold = 0.0000000001
)
{
    return std::abs(a_x_0 - a_x_1) < a_threshold;
}

void test_tensor_default_constructor(

)
{
    constexpr size_t VECTOR_SIZE = 10000;
    constexpr size_t MATRIX_ROWS = 400;
    constexpr size_t MATRIX_COLS = 30;

    tensor<double, VECTOR_SIZE> l_tens_0 = constant<double, VECTOR_SIZE>(0);
    
    for (const double& l_double : l_tens_0)
        // Check for default construction of elements.
        assert(is_close_to(l_double, 0));


    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tens_1 = constant<double, MATRIX_ROWS, MATRIX_COLS>(0);

    for (const tensor<double, MATRIX_COLS>& l_row : l_tens_1)
        for (const double& l_double : l_row)
            // Check for default construction of elements.
            assert(is_close_to(l_double, 0));

}

void test_make_state_gradient_pair(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 300;

    model::begin();

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_0_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(is_close_to(l_tens_0_ptr[i][j]->m_state, 0.0));

    model l_model = model::end();

}

void test_get_state(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;

    model::begin();

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_0_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    // Just initialize all of the values in the matrix to be different.
    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            l_tens_0_ptr[i][j]->m_state = i * MATRIX_COLS + j;
    
    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tens_0_state = get_state(l_tens_0_ptr);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(is_close_to(l_tens_0_state[i][j], l_tens_0_ptr[i][j]->m_state));
    
    model l_model = model::end();
    
}

void test_set_state(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 300;

    model::begin();
    
    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_0_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tens_0_state;

    // Initialize the l_tens_0_state to non-default values
    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            l_tens_0_state[i][j] = i * MATRIX_COLS + j;

    set_state(l_tens_0_ptr, l_tens_0_state);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(is_close_to(l_tens_0_ptr[i][j]->m_state, l_tens_0_state[i][j]));

    model l_model = model::end();

}

void test_get_gradient(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 300;

    model::begin();

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_0_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            state_gradient_pair_dependency l_dep = l_tens_0_ptr[i][j]->depend();
            l_dep.m_partial_gradient = i * MATRIX_COLS + j;
        }

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tens_0_grad = get_gradient(l_tens_0_ptr);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(is_close_to(l_tens_0_grad[i][j], i * MATRIX_COLS + j));

    model l_model = model::end();

}

void test_partition(

)
{
    constexpr size_t MATRIX_ROWS = 300;
    constexpr size_t MATRIX_COLS = 100;
    constexpr size_t VALID_VECTOR_DIMS   = MATRIX_ROWS * MATRIX_COLS;

    tensor<double, VALID_VECTOR_DIMS> l_valid_size_vector = constant<double, VALID_VECTOR_DIMS>(0);

    for (int i = 0; i < VALID_VECTOR_DIMS; i++)
        l_valid_size_vector[i] = i;

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_partitioned_valid_size_vector = partition<MATRIX_ROWS>(l_valid_size_vector);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(is_close_to(l_partitioned_valid_size_vector[i][j], l_valid_size_vector[i * MATRIX_COLS + j]));

}

void test_concat(

)
{
    constexpr size_t MATRIX_0_ROWS = 300;
    constexpr size_t MATRIX_0_COLS = 100;
    constexpr size_t MATRIX_1_ROWS = 400;
    constexpr size_t MATRIX_1_COLS = 100;

    tensor<double, MATRIX_0_ROWS, MATRIX_0_COLS> l_tens_0 = constant<double, MATRIX_0_ROWS, MATRIX_0_COLS>(0);

    for (int i = 0; i < MATRIX_0_ROWS; i++)
        for (int j = 0; j < MATRIX_0_COLS; j++)
            l_tens_0[i][j] = i * MATRIX_0_COLS + j;

    tensor<double, MATRIX_1_ROWS, MATRIX_1_COLS> l_tens_1 = constant<double, MATRIX_1_ROWS, MATRIX_1_COLS>(0);

    for (int i = 0; i < MATRIX_1_ROWS; i++)
        for (int j = 0; j < MATRIX_1_COLS; j++)
            l_tens_1[i][j] = i * MATRIX_1_COLS + j + MATRIX_0_ROWS * MATRIX_0_COLS;

    tensor<double, MATRIX_0_ROWS + MATRIX_1_ROWS, MATRIX_0_COLS> l_concatenated = concat(l_tens_0, l_tens_1);

    for (int i = 0; i < MATRIX_0_ROWS + MATRIX_1_ROWS; i++)
        for (int j = 0; j < MATRIX_0_COLS; j++)
            assert(is_close_to(l_concatenated[i][j], i * MATRIX_0_COLS + j));

}

void test_flatten(

)
{
    constexpr size_t TENSOR_DEPTH = 100;
    constexpr size_t TENSOR_HEIGHT = 20;
    constexpr size_t TENSOR_WIDTH = 30;
    
    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0 = constant<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH>(0);

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
                l_tens_0[i][j][k] = i * TENSOR_HEIGHT * TENSOR_WIDTH + j * TENSOR_WIDTH + k;

    tensor<double, TENSOR_DEPTH * TENSOR_HEIGHT * TENSOR_WIDTH> l_flattened = flatten(l_tens_0);

    for (int i = 0; i < l_flattened.size(); i++)
        assert(is_close_to(l_flattened[i], l_tens_0[i / (TENSOR_HEIGHT * TENSOR_WIDTH)][(i / TENSOR_WIDTH) % TENSOR_HEIGHT][i % TENSOR_WIDTH]));

}

void test_add(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    
    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_1;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_1[i][j][k] = (i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k) % 15;
                l_expected[i][j][k] = l_tens_0[i][j][k] + l_tens_1[i][j][k];
            }

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_added = add(l_tens_0, l_tens_1);

    assert(l_added == l_expected);
    
}

void test_additive_aggregate(

)
{
    constexpr size_t VECTOR_SIZE = 10;
    constexpr size_t MATRIX_ROWS = 300;
    constexpr size_t MATRIX_COLS = 10;
    
    tensor<double, VECTOR_SIZE> l_tens_0;

    double l_expected_0 = 0;

    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        l_tens_0[i] = i;
        l_expected_0 += i;
    }

    double l_additive_aggregate_0 = additive_aggregate(l_tens_0);

    assert(l_additive_aggregate_0 == l_expected_0);

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tens_1;

    tensor<double, MATRIX_COLS> l_expected_1 = constant<double, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_tens_1[i][j] = i * MATRIX_COLS + j;
            l_expected_1[j] += i * MATRIX_COLS + j;
        }

    tensor<double, MATRIX_COLS> l_additive_aggregate_1 = additive_aggregate(l_tens_1);

    assert(l_additive_aggregate_1 == l_expected_1);

}

void test_subtract(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    
    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_1;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_1[i][j][k] = (i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k) % 15;
                l_expected[i][j][k] = l_tens_0[i][j][k] - l_tens_1[i][j][k];
            }

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_subtracted = subtract(l_tens_0, l_tens_1);

    assert(l_subtracted == l_expected);
    
}

void test_tensor_tensor_multiply(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    
    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_1;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_1[i][j][k] = (i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k) % 15;
                l_expected[i][j][k] = l_tens_0[i][j][k] * l_tens_1[i][j][k];
            }

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_multiplied = multiply(l_tens_0, l_tens_1);

    assert(l_multiplied == l_expected);
    
}

void test_tensor_scalar_multiply(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    constexpr double SCALAR = 10.3;
    
    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;
                l_expected[i][j][k] = l_tens_0[i][j][k] * SCALAR;
            }

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_multiplied = multiply(l_tens_0, SCALAR);

    assert(l_multiplied == l_expected);

}

void test_average(

)
{
    constexpr size_t VECTOR_SIZE = 10;
    constexpr size_t MATRIX_ROWS = 300;
    constexpr size_t MATRIX_COLS = 10;
    
    tensor<double, VECTOR_SIZE> l_tens_0;

    double l_expected_0 = 0;

    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        l_tens_0[i] = i;
        l_expected_0 += i;
    }

    l_expected_0 /= (double)VECTOR_SIZE;

    double l_average_0 = average(l_tens_0);

    assert(l_average_0 == l_expected_0);

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tens_1;

    tensor<double, MATRIX_COLS> l_expected_1 = constant<double, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_tens_1[i][j] = i * MATRIX_COLS + j;
            l_expected_1[j] += i * MATRIX_COLS + j;
        }

    l_expected_1 = multiply(l_expected_1, 1.0 / (double)MATRIX_ROWS);

    tensor<double, MATRIX_COLS> l_average_1 = average(l_tens_1);

    assert(l_average_1 == l_expected_1);

}

void test_row(

)
{
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    constexpr size_t ROW_INDEX = 2;
    
    tensor<double, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_HEIGHT; i++)
    {
        for (int j = 0; j < TENSOR_WIDTH; j++)
            l_tens_0[i][j] = i * TENSOR_WIDTH + j;
        if (i == ROW_INDEX)
        {
            l_expected = l_tens_0[i];
        }
    }

    tensor<double, TENSOR_WIDTH> l_row = row(l_tens_0, ROW_INDEX);

    assert(l_row == l_expected);

}

void test_col(

)
{
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    constexpr size_t COL_INDEX = 7;
    
    tensor<double, TENSOR_HEIGHT> l_expected;

    tensor<double, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_HEIGHT; i++)
    {
        for (int j = 0; j < TENSOR_WIDTH; j++)
        {
            l_tens_0[i][j] = i * TENSOR_WIDTH + j;

            if (j == COL_INDEX)
            {
                l_expected[i] = l_tens_0[i][j];
            }

        }
    }

    tensor<double, TENSOR_HEIGHT> l_col = col(l_tens_0, COL_INDEX);

    assert(l_col == l_expected);

}

void test_vector_dot(

)
{
    constexpr size_t VECTOR_SIZE = 14;

    tensor<double, VECTOR_SIZE> l_tens_0;
    tensor<double, VECTOR_SIZE> l_tens_1;
    
    double l_expected = 0;

    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        l_tens_0[i] = i;
        l_tens_1[i] = i % 2;
        l_expected += l_tens_0[i] * l_tens_1[i];
    }

    double l_dot = dot(l_tens_0, l_tens_1);

    assert(l_dot == l_expected);

}

void test_matrix_dot(

)
{
    constexpr size_t MATRIX_0_ROWS = 10;
    constexpr size_t MATRIX_0_COLS = 4;
    constexpr size_t MATRIX_1_COLS = 14;

    tensor<double, MATRIX_0_ROWS, MATRIX_0_COLS> l_tensor_0;

    for (int i = 0; i < MATRIX_0_ROWS; i++)
        for (int j = 0; j < MATRIX_0_COLS; j++)
            l_tensor_0[i][j] = i * MATRIX_0_COLS + j;

    tensor<double, MATRIX_0_COLS, MATRIX_1_COLS> l_tensor_1;

    for (int i = 0; i < MATRIX_0_COLS; i++)
        for (int j = 0; j < MATRIX_1_COLS; j++)
            l_tensor_1[i][j] = i * MATRIX_1_COLS + j;

    tensor<double, MATRIX_0_ROWS, MATRIX_1_COLS> l_expected = constant<double, MATRIX_0_ROWS, MATRIX_1_COLS>(0);

    for (int i = 0; i < MATRIX_0_ROWS; i++)
        for (int j = 0; j < MATRIX_1_COLS; j++)
            for (int k = 0; k < MATRIX_0_COLS; k++)
                l_expected[i][j] += l_tensor_0[i][k] * l_tensor_1[k][j];

    tensor<double, MATRIX_0_ROWS, MATRIX_1_COLS> l_dot = dot(l_tensor_0, l_tensor_1);

    assert(l_dot == l_expected);

}

void test_transpose(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    
    tensor<double, TENSOR_HEIGHT, TENSOR_DEPTH, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;
                l_expected[j][i][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;
            }

    tensor<double, TENSOR_HEIGHT, TENSOR_DEPTH, TENSOR_WIDTH> l_transposed = transpose(l_tens_0);

    assert(l_transposed == l_expected);

}

void test_negate(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    constexpr double SCALAR = -1;
    
    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_expected;

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;
                l_expected[i][j][k] = l_tens_0[i][j][k] * SCALAR;
            }

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_multiplied = negate(l_tens_0);

    assert(l_multiplied == l_expected);

}

void test_io(

)
{
    constexpr size_t TENSOR_DEPTH = 3;
    constexpr size_t TENSOR_HEIGHT = 100;
    constexpr size_t TENSOR_WIDTH = 10;
    const std::string FILE_PATH = "test_tensor_io.bin";

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_tens_0;

    for (int i = 0; i < TENSOR_DEPTH; i++)
        for (int j = 0; j < TENSOR_HEIGHT; j++)
            for (int k = 0; k < TENSOR_WIDTH; k++)
            {
                l_tens_0[i][j][k] = i * (TENSOR_HEIGHT * TENSOR_WIDTH) + j * TENSOR_WIDTH + k;
            }

    std::ofstream l_ofs(FILE_PATH);

    l_ofs << l_tens_0;
    
    l_ofs.close();

    tensor<double, TENSOR_DEPTH, TENSOR_HEIGHT, TENSOR_WIDTH> l_recovered;

    std::ifstream l_ifs(FILE_PATH);

    l_ifs >> l_recovered;

    l_ifs.close();

    assert(l_recovered == l_tens_0);

    std::filesystem::remove(FILE_PATH);

}

void test_add_state_gradient_pairs(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    
    model::begin();

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected_states;

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_0_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);
    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_1_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_tens_0_ptr[i][j]->m_state = i * MATRIX_COLS + j;
            l_tens_1_ptr[i][j]->m_state = (i * MATRIX_COLS + j) % 5;
            l_expected_states[i][j] = l_tens_0_ptr[i][j]->m_state + l_tens_1_ptr[i][j]->m_state;
        }

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_added = add(l_tens_0_ptr, l_tens_1_ptr);

    model l_model = model::end();

    l_model.fwd();

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states = get_state(l_added);

    assert(l_states == l_expected_states);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            l_added[i][j]->depend().m_partial_gradient = i * MATRIX_COLS + j;

    l_model.bwd();

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            assert(l_tens_0_ptr[i][j]->gradient() == i * MATRIX_COLS + j);
            assert(l_tens_1_ptr[i][j]->gradient() == i * MATRIX_COLS + j);
        }
    
}

void test_subtract_state_gradient_pairs(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    
    model::begin();

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected_states;

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_0_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);
    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_tens_1_ptr = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_tens_0_ptr[i][j]->m_state = i * MATRIX_COLS + j;
            l_tens_1_ptr[i][j]->m_state = (i * MATRIX_COLS + j) % 5;
            l_expected_states[i][j] = l_tens_0_ptr[i][j]->m_state - l_tens_1_ptr[i][j]->m_state;
        }

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_subtracted = subtract(l_tens_0_ptr, l_tens_1_ptr);

    model l_model = model::end();

    l_model.fwd();

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states = get_state(l_subtracted);

    assert(l_states == l_expected_states);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            l_subtracted[i][j]->depend().m_partial_gradient = i * MATRIX_COLS + j;

    l_model.bwd();

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            assert(l_tens_0_ptr[i][j]->gradient() ==   i * MATRIX_COLS + j);
            assert(l_tens_1_ptr[i][j]->gradient() == -(i * (int)MATRIX_COLS + j));
        }
    
}

void test_sigmoid()
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    
    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states;

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected;

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j] = i * 0.1 + j * 0.01;
            l_expected[i][j] = sigmoid(l_states[i][j]);
        }

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_sigmoid = sigmoid(l_states);

    assert(l_sigmoid == l_expected);

}

void test_tanh(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    
    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states;

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected;


    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j] = i * 0.1 + j * 0.01 - 10;
            l_expected[i][j] = tanh(l_states[i][j]);
        }

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_tanh = tanh(l_states);

    assert(l_tanh == l_expected);

}

void test_leaky_relu(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    
    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states;

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected;


    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j] = i * 0.1 + j * 0.01 - 10;
            l_expected[i][j] = leaky_relu(l_states[i][j], 0.3);
        }

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_leaky_relu = leaky_relu(l_states, 0.3);

    assert(l_leaky_relu == l_expected);
    
}

void test_log(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    
    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states;

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected;


    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j] = i * 0.1 + j * 0.01 + 0.01;
            l_expected[i][j] = log(l_states[i][j]);
        }

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_log = log(l_states);

    assert(l_log == l_expected);

}

void test_pow(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    constexpr double POWER = 2.0;
    
    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_states;

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected;


    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j] = i * 0.1 + j * 0.01 + 0.01;
            l_expected[i][j] = pow(l_states[i][j], POWER);
        }

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_pow = pow(l_states, POWER);

    assert(l_pow == l_expected);

}

void test_pow_state_gradient_pair(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    constexpr double POWER = 2;

    model::begin();

    state_gradient_pair l_power(POWER);

    state_gradient_pair* l_power_ptr = &l_power;

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_states = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    tensor<double, MATRIX_ROWS, MATRIX_COLS> l_expected;

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j]->m_state = i * 0.1 + j * 0.01 + 0.01;
            l_expected[i][j] = pow(l_states[i][j]->m_state, POWER);
        }

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_pow = pow(l_states, l_power_ptr);

    model l_model = model::end();

    l_model.fwd();

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(l_pow[i][j]->m_state == l_expected[i][j]);

}

void test_insertion_extraction_operators(

)
{
    constexpr size_t MATRIX_ROWS = 100;
    constexpr size_t MATRIX_COLS = 30;
    constexpr double POWER = 2;

    model::begin();

    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_states = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
        {
            l_states[i][j]->m_state = i * 0.1 + j * 0.01 + 0.01;
        }
    
    tensor<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS> l_recovered = constant<state_gradient_pair*, MATRIX_ROWS, MATRIX_COLS>(0);

    model l_model = model::end();

    std::stringstream l_ss;

    l_ss << l_states;

    l_ss >> l_recovered;

    for (int i = 0; i < MATRIX_ROWS; i++)
        for (int j = 0; j < MATRIX_COLS; j++)
            assert(is_close_to(l_states[i][j]->m_state, l_recovered[i][j]->m_state));

}

void test_concat_more_than_two(

)
{
    constexpr size_t MATRIX_0_ROWS = 300;
    constexpr size_t MATRIX_1_ROWS = 400;
    constexpr size_t MATRIX_2_ROWS = 200;

    constexpr size_t MATRIX_COMMON_COLS = 100;

    tensor<double, MATRIX_0_ROWS, MATRIX_COMMON_COLS> l_tens_0 = constant<double, MATRIX_0_ROWS, MATRIX_COMMON_COLS>(0);

    for (int i = 0; i < MATRIX_0_ROWS; i++)
        for (int j = 0; j < MATRIX_COMMON_COLS; j++)
            l_tens_0[i][j] = i * MATRIX_COMMON_COLS + j;

    tensor<double, MATRIX_1_ROWS, MATRIX_COMMON_COLS> l_tens_1 = constant<double, MATRIX_1_ROWS, MATRIX_COMMON_COLS>(0);

    for (int i = 0; i < MATRIX_1_ROWS; i++)
        for (int j = 0; j < MATRIX_COMMON_COLS; j++)
            l_tens_1[i][j] = i * MATRIX_COMMON_COLS + j + MATRIX_0_ROWS * MATRIX_COMMON_COLS;

    tensor<double, MATRIX_2_ROWS, MATRIX_COMMON_COLS> l_tens_2 = constant<double, MATRIX_2_ROWS, MATRIX_COMMON_COLS>(0);

    for (int i = 0; i < MATRIX_2_ROWS; i++)
        for (int j = 0; j < MATRIX_COMMON_COLS; j++)
            l_tens_2[i][j] = i * MATRIX_COMMON_COLS + j + MATRIX_0_ROWS * MATRIX_COMMON_COLS + MATRIX_1_ROWS * MATRIX_COMMON_COLS;

    tensor<double, MATRIX_0_ROWS + MATRIX_1_ROWS + MATRIX_2_ROWS, MATRIX_COMMON_COLS> l_concatenated = concat(l_tens_0, l_tens_1, l_tens_2);

    for (int i = 0; i < MATRIX_0_ROWS + MATRIX_1_ROWS + MATRIX_2_ROWS; i++)
        for (int j = 0; j < MATRIX_COMMON_COLS; j++)
            assert(is_close_to(l_concatenated[i][j], i * MATRIX_COMMON_COLS + j));

}

void test_flatten_concat(

)
{
    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

    auto l_tensor_0 = constant<double, 100>(l_randomly_generate_parameter);
    auto l_tensor_1 = constant<double, 30, 20>(l_randomly_generate_parameter);
    auto l_tensor_2 = constant<double, 40, 5, 2>(l_randomly_generate_parameter);

    auto l_concatenated = flatten(l_tensor_0, l_tensor_1, l_tensor_2);

    int it = 0;

    for (int i = 0; i < l_tensor_0.size(); i++, it++)
        assert(is_close_to(l_tensor_0[i], l_concatenated[it]));

    for (int i = 0; i < l_tensor_1.size(); i++)
        for (int j = 0; j < l_tensor_1[0].size(); j++, it++)
            assert(is_close_to(l_tensor_1[i][j], l_concatenated[it]));

    for (int i = 0; i < l_tensor_2.size(); i++)
        for (int j = 0; j < l_tensor_2[0].size(); j++)
            for (int k = 0; k < l_tensor_2[0][0].size(); k++, it++)
                assert(is_close_to(l_tensor_2[i][j][k], l_concatenated[it]));

}

void test_tensor_copy(

)
{
    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

    auto l_tens_0 = constant<double, 10, 10>(l_randomly_generate_parameter);
    auto l_tens_1 = constant<double, 100>();
    auto l_tens_2 = constant<double, 2, 5, 10>();

    auto l_tens_3 = constant<double, 1, 5, 10>();
    auto l_tens_4 = constant<double, 2, 5, 5>();

    copy(l_tens_0, l_tens_1);
    copy(l_tens_0, l_tens_2);    
    copy(l_tens_0, l_tens_3, l_tens_4);

    auto l_tens_0_flattened = flatten(l_tens_0);
    auto l_tens_1_flattened = flatten(l_tens_1);
    auto l_tens_2_flattened = flatten(l_tens_2);
    auto l_tens_3_tens_4_concat = flatten(l_tens_3, l_tens_4);

    assert(l_tens_1_flattened == l_tens_0_flattened);
    assert(l_tens_2_flattened == l_tens_0_flattened);
    assert(l_tens_3_tens_4_concat == l_tens_0_flattened);

}

enum class inclusion_mode : uint8_t
{
    EXCLUSION = 0,
    POSITIVE_INCLUSION = 1,
    NEGATIVE_INCLUSION = 2,
};

template<size_t BLOCK_SIZE, size_t VARIABLE_COUNT>
class additive_subtractive_model
{
    tensor<inclusion_mode, BLOCK_SIZE, VARIABLE_COUNT> m_terms;

    std::function<bool(const tensor<bool, VARIABLE_COUNT>&)> m_get_internal_state;

    bool m_additive_mode;

public:
    additive_subtractive_model(
        const tensor<inclusion_mode, BLOCK_SIZE, VARIABLE_COUNT>& a_terms,
        const bool& a_additive_mode,
        const std::function<bool(const tensor<bool, VARIABLE_COUNT>&)>& a_get_internal_state
    ) :
        m_terms(a_terms),
        m_get_internal_state(a_get_internal_state),
        m_additive_mode(a_additive_mode)
    {

    }

    additive_subtractive_model(
        const additive_subtractive_model& a_other
    ) :
        m_terms(a_other.m_terms),
        m_get_internal_state(a_other.m_get_internal_state),
        m_additive_mode(a_other.m_additive_mode)
    {

    }

    bool evaluate(
        const tensor<bool, VARIABLE_COUNT>& a_x
    ) const
    {
        bool l_internal_state = m_get_internal_state(a_x);

        if (m_additive_mode)
        {
            if (l_internal_state)
                // Since we are in additive mode, satisfaction of the internal state implies
                // satisfaction of the composite state.
                return true;

            return sum_covers(a_x);
            
        }
        else
        {
            if (!l_internal_state)
                // Since we are in subtractive mode, dissatisfaction of the internal state implies
                // dissatisfaction of the composite state.
                return false;

            return !sum_covers(a_x);
                
        }
        
    }

private:
    bool sum_covers(
        const tensor<bool, VARIABLE_COUNT>& a_x
    ) const
    {
        return std::any_of(
            m_terms.begin(),
            m_terms.end(),
            [&a_x](
                const tensor<inclusion_mode, VARIABLE_COUNT>& a_term
            )
            {
                return product_covers(a_term, a_x);
            }
        );
    }

    static bool product_covers(
        const tensor<inclusion_mode, VARIABLE_COUNT>& a_term,
        const tensor<bool, VARIABLE_COUNT>& a_x
    )
    {
        for (int i = 0; i < VARIABLE_COUNT; i++)
        {
            if (a_term[i] == inclusion_mode::POSITIVE_INCLUSION && !a_x[i])
                return false;

            if (a_term[i] == inclusion_mode::NEGATIVE_INCLUSION && a_x[i])
                return false;

            // ignored case: inclusion_mode::EXCLUSION
            
        }

        return true;
        
    }
    
};

template<size_t I, typename INT_TYPE>
tensor<bool, I> get_bits(
    const INT_TYPE& a_int
)
{
    tensor<bool, I> l_result;

    for (int i = 0; i < I; i++)
        l_result[i] = ((0x1 << i) & a_int) != 0;
    
    return l_result;
    
}

void test_additive_subtractive_model(

)
{
    constexpr size_t VARIABLE_COUNT = 8;
    constexpr size_t BLOCK_SIZE = 10;

    constexpr size_t STAGNANT_EPOCH_COUNT_BEFORE_PROCEEDING = 100000;
    constexpr size_t BLOCK_COUNT = 10;

    constexpr size_t PARTICLE_COUNT = 30;
    constexpr size_t TRAINING_SAMPLE_SIZE = 100;
    constexpr size_t TESTING_SAMPLE_SIZE = 10000;

    constexpr size_t PARAMETER_COUNT = BLOCK_SIZE * VARIABLE_COUNT;

    // Initialize additive_subtractive_model args.
    bool l_additive_mode = true;
    std::function<bool(const tensor<bool, VARIABLE_COUNT>&)> l_get_internal_state = [](
        const tensor<bool, VARIABLE_COUNT>& a_x
    )
    {
        return false;
    };

    // Define the reward function in terms of the sum.
    auto l_get_accuracy = [&l_additive_mode, &l_get_internal_state](
        const tensor<inclusion_mode, PARAMETER_COUNT>& a_parameter_vector,
        const size_t& a_sample_size
    )
    {
        additive_subtractive_model<BLOCK_SIZE, VARIABLE_COUNT> l_asm(
            partition<BLOCK_SIZE>(a_parameter_vector),
            l_additive_mode,
            l_get_internal_state
        );

        size_t l_equivalence_coverage = 0;

        // We are trying to model 4-bit integer addition.

        for (int i = 0; i < a_sample_size; i++)
        {
            uint8_t l_int_0 = rand() % 16;
            uint8_t l_int_1 = rand() % 16;
            uint8_t l_sum = l_int_0 + l_int_1;
            auto l_sum_bits = get_bits<8>(l_sum);

            bool l_desired_boolean_output = l_sum_bits[3] == l_sum_bits[1];

            l_equivalence_coverage += l_asm.evaluate(concat(get_bits<4>(l_int_0), get_bits<4>(l_int_1))) == l_desired_boolean_output;

        }

        return (double)l_equivalence_coverage / (double)a_sample_size;

    };

    // Create icpso optimizer instance
    icpso<uint8_t, PARTICLE_COUNT, PARAMETER_COUNT> l_icpso(
        constant<size_t, PARAMETER_COUNT>(3),
        0.9, 0.1, 0.1, 0.6, 0.5
    );

    for (int l_block_index = 0; l_block_index < BLOCK_COUNT; l_block_index++)
    {
        double l_best_accuracy = 0;
        int l_last_epoch_of_improvement = 0;
        
        // Train using icpso for a set amount of epochs
        for (
            int l_epoch_index = 0;
            (l_epoch_index - l_last_epoch_of_improvement) < STAGNANT_EPOCH_COUNT_BEFORE_PROCEEDING;
            l_epoch_index++
        )
        {
            auto l_candidate_solutions = convert<inclusion_mode>(l_icpso.candidate_solutions());
            
            tensor<double, PARTICLE_COUNT> l_particle_rewards;

            // Populate the l_particle_rewards vector with the rewards of each particle.
            std::transform(
                l_candidate_solutions.begin(),
                l_candidate_solutions.end(),
                l_particle_rewards.begin(),
                [&l_get_accuracy, &TRAINING_SAMPLE_SIZE](
                    const tensor<inclusion_mode, PARAMETER_COUNT>& a_parameter_vector
                )
                {
                    return l_get_accuracy(a_parameter_vector, TRAINING_SAMPLE_SIZE);
                }
            );

            l_icpso.update(l_particle_rewards);

            if (l_epoch_index % 100 == 0)
                std::cout << "BLOCK: " << l_block_index << (l_additive_mode? " (ADDITIVE)" : " (SUBTRACTIVE)") << ", EPOCH: " << l_epoch_index << ", ACCURACY: " << l_icpso.global_best_reward() << std::endl;

            if (l_icpso.global_best_reward() > l_best_accuracy)
            {
                l_best_accuracy = l_icpso.global_best_reward();
                l_last_epoch_of_improvement = l_epoch_index;
            }

        }


        // Create an additive_subtractive_model with the optimal parameter vector,
        // and save it by capture copy in a lambda expression.
        additive_subtractive_model<BLOCK_SIZE, VARIABLE_COUNT> l_asm(
            partition<BLOCK_SIZE>(convert<inclusion_mode>(l_icpso.global_best_solution())),
            l_additive_mode,
            l_get_internal_state
        );

        
        l_get_internal_state = [l_asm](
            const tensor<bool, VARIABLE_COUNT>& a_x
        )
        {
            return l_asm.evaluate(a_x);
        };

        // Switch mode from additive to subtractive and vice versa.
        l_additive_mode = !l_additive_mode;

        // Reset the optimization session.
        l_icpso.reset();

    }
    
}

void unit_test_main(

)
{
    test_tensor_default_constructor();
    test_make_state_gradient_pair();
    test_get_state();
    test_set_state();
    test_get_gradient();
    test_partition();
    test_concat();
    test_flatten();
    test_add();
    test_additive_aggregate();
    test_subtract();
    test_tensor_tensor_multiply();
    test_tensor_scalar_multiply();
    test_average();
    test_row();
    test_col();
    test_vector_dot();
    test_matrix_dot();
    test_transpose();
    test_negate();
    test_io();
    test_add_state_gradient_pairs();
    test_subtract_state_gradient_pairs();
    test_sigmoid();
    test_tanh();
    test_leaky_relu();
    test_log();
    test_pow();
    test_pow_state_gradient_pair();
    test_insertion_extraction_operators();
    test_concat_more_than_two();
    test_flatten_concat();
    test_tensor_copy();
    tnn_test();
    parabola_test();
    lstm_test();
    large_memory_usage_test();
    test_pso();
    test_icpso();
}

void test_waveform_path_tracing(

)
{
    auto I = [](const double& a_x){ return -a_x; };
    auto O = [](const std::vector<double>& a_vec){ return std::accumulate(a_vec.begin(), a_vec.end(), 0.0); };
    auto A = [](const std::vector<double>& a_vec){ return std::accumulate(a_vec.begin(), a_vec.end(), 1.0, [](const double& a_x1, const double& a_x2){ return a_x1*a_x2; }); };
    
    auto a = [](const double& a_x){ return sin(M_PI/2.0*a_x); };
    auto b = [](const double& a_x){ return sin(M_PI/3.0*a_x); };
    auto c = [](const double& a_x){ return sin(M_PI/5.0*a_x); };
    auto d = [](const double& a_x){ return sin(M_PI/7.0*a_x); };
    auto e = [](const double& a_x){ return sin(M_PI/11.0*a_x); };
    auto f = [](const double& a_x){ return sin(M_PI/13.0*a_x); };

    auto f1 = a;
    auto f2 = [&](const double& x){ 
        return 
            f1(x) / I(a(x)) * A({ I(b(x)), I(c(x)) }) +
            f1(x) / a(x)    * O({ A({ I(b(x)), c(x) }), A({ b(x), I(c(x)) }), A({ b(x), c(x) }) });
    };
    
    auto f3 = [&](const double& x){ 
        return 
            f2(x) / I(b(x)) * O({ A({I(d(x)), I(e(x))}), A({I(d(x)), e(x)}), A({d(x), I(e(x))}) }) +
            f2(x) / b(x)    * A({ d(x), e(x) });
    };
    
    auto f4 = [&](const double& x){
        return
            f2(x) / I(c(x)) * f(x) +
            f2(x) / c(x)    * I(f(x));
    };

    auto f_combined = [&](const double& x)
    {
        return O({ f3(x), f4(x) });
    };

    constexpr double WINDOW_SIZE = 10;
    constexpr double STEP_SIZE = 0.1;

    constexpr size_t SAMPLE_SIZE = (size_t)(ceil(WINDOW_SIZE / STEP_SIZE));

    tensor<double, SAMPLE_SIZE> l_unshifted_ys;
    tensor<double, SAMPLE_SIZE> l_shifted_ys;

    constexpr double MAXIMUM_ACCEPTABLE_MSE = 0.1;

    for (int shift = 0; true; shift++)
    {
        // Populate the tensors before computing MSE
        for (int i = 0; i < SAMPLE_SIZE; i++)
        {
            double x = -WINDOW_SIZE/2.0 + (double)i * STEP_SIZE;
            l_unshifted_ys[i] = f_combined(x);
            l_shifted_ys[i]   = f_combined(x - shift);
        }

        if (mean_squared_error(l_unshifted_ys, l_shifted_ys) <= MAXIMUM_ACCEPTABLE_MSE)
        {
            std::cout << "PERIODICITY FOUND: " << shift << std::endl;
            std::cin.get();
        }

        if (shift % 100 == 0)
            std::cout << "SHIFT TESTED: " << shift << std::endl;
        
    }
    
}

double derivative(
    const std::function<double(double)>& a_f,
    const double& a_center,
    const size_t& a_order = 1,
    const double& a_h = 0.0001
)
{
    if (a_order == 0)
    {
        // Recursion ends here
        return a_f(a_center);
    }
    else
    {
        double l_lower_order_derivative_0 = derivative(a_f, a_center - a_h, a_order - 1, a_h);
        double l_lower_order_derivative_1 = derivative(a_f, a_center + a_h, a_order - 1, a_h);
        return (l_lower_order_derivative_1 - l_lower_order_derivative_0) / (2.0 * a_h);
    }
}

void train_localized_multiplicative_distributor(

)
{
    // MODEL INPUTS
    // t, f, f', f'', f''', g, g', g'', g'''

    constexpr size_t MODEL_INPUT_DIMS = 9;
    constexpr size_t MODEL_OUTPUT_DIMS = 1;
    constexpr std::array<size_t, 2> MODEL_LAYER_DIMS = { 32, 20 };
    
    // SET UP PARAMETER VALUE GENERATION
    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-0.1, 0.1);
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };

#pragma region CONSTRUCTION OF MODEL

    model::begin();

    auto l_x = input<MODEL_INPUT_DIMS>();
    auto l_desired_y = input<MODEL_OUTPUT_DIMS>();

    auto l_w0_y = multiply(input<MODEL_LAYER_DIMS[0], MODEL_INPUT_DIMS>(l_randomly_generate_parameter), l_x);
    auto l_b0_y = add(input<MODEL_LAYER_DIMS[0]>(l_randomly_generate_parameter), l_w0_y);
    auto l_layer_0_y = leaky_relu(l_w0_y, 0.3);
    
    auto l_w1_y = multiply(input<MODEL_LAYER_DIMS[1], MODEL_LAYER_DIMS[0]>(l_randomly_generate_parameter), l_layer_0_y);
    auto l_b1_y = add(input<MODEL_LAYER_DIMS[1]>(l_randomly_generate_parameter), l_w1_y);
    auto l_layer_1_y = leaky_relu(l_b1_y, 0.3);

    auto l_w2_y = multiply(input<MODEL_OUTPUT_DIMS, MODEL_LAYER_DIMS[1]>(l_randomly_generate_parameter), l_layer_1_y);
    auto l_b2_y = add(input<MODEL_OUTPUT_DIMS>(l_randomly_generate_parameter), l_w2_y);
    auto l_y = leaky_relu(l_b2_y, 0.3);

    auto l_loss = mean_squared_error(l_y, l_desired_y)->depend();

    model l_model = model::end();

#pragma endregion

    // COLLECT ALL PARAMETERS INTO A SINGLE VECTOR
    auto l_parameter_vector = flatten(
        l_w0_y,
        l_b0_y,
        l_w1_y,
        l_b1_y,
        l_w2_y,
        l_b2_y
    );

    // LAMBDA FUNCTION WHICH CREATES A RANDOM SET OF PRIMES
    auto l_generate_random_set_of_primes = []
    {
        constexpr std::array<int, 10> l_primes =
        {
            3, 5, 7, 11, 13, 17, 19, 23, 29, 31
        };
        
        std::set<double> l_result;

        size_t l_sample_size = rand() % l_primes.size() + 1;

        for (int i = 0; i < l_sample_size; i++)
        {
            l_result.insert(l_primes[rand() % l_primes.size()]);
        }

        return l_result;
        
    };

    // MANUALLY COMPUTES A MULTIPLICATIVE DISTRIBUTION OF TWO SUMS
    auto l_multiplicative_distribute = [](
        const std::set<double>& a_set_0,
        const std::set<double>& a_set_1
    )
    {
        std::set<double> l_result;

        for (const double& l_double_0 : a_set_0)
            for (const double& l_double_1 : a_set_1)
                l_result.insert(l_double_0 * l_double_1);

        return l_result;
        
    };

    // MANUALLY EVALUATES A WAVEFORM OVER AN INPUT VALUE
    auto l_evaluate_waveform = [](
        const std::set<double>& a_constituent_wave_periods,
        const double& a_x
    )
    {
        // ASSUME SIN FUNCTIONS ONLY.
        double l_result = 0;

        for (const double& l_period : a_constituent_wave_periods)
            l_result += sin(2.0 * M_PI / l_period * a_x);
        
        return l_result;
        
    };

    auto l_waveform_derivative = [&](
        const std::set<double>& a_constituent_wave_periods,
        const double& a_center,
        const size_t& a_order
    )
    {
        return derivative(
            [&](const double& a_x)
            {
                return l_evaluate_waveform(a_constituent_wave_periods, a_x); 
            },
            a_center,
            a_order
        );
    };

    // LAMBDA FUNCTION WHICH:
    //  1. GENERATES 2 SETS OF PRIMES
    //  2. MANUALLY COMPUTES THE DESIRED WAVEFORM
    //  3. EVALUATES THE MODEL AGAINST THE DESIRED WAVEFORM.
    auto l_evaluate_loss = [&]
    {
        static std::uniform_real_distribution<double> l_domain_random(-10000, 10000);

        // GENERATE TWO SETS OF PRIMES FOR USE AS OPERANDS OF COMBINATION
        std::set<double> l_primes_0 = l_generate_random_set_of_primes();
        std::set<double> l_primes_1 = l_generate_random_set_of_primes();

        // MANUALLY COMPUTE THE DISTRIBUTION
        std::set<double> l_distribution = l_multiplicative_distribute(l_primes_0, l_primes_1);

        constexpr size_t PREDICTION_COUNT = 50;

        double l_average_loss = 0;

        for (int i = 0; i < PREDICTION_COUNT; i++)
        {
            double l_random_x = l_domain_random(l_dre);

            set_state(l_x, 
            {
                l_random_x / 10000.0,
                l_waveform_derivative(l_primes_0, l_random_x, 0),
                l_waveform_derivative(l_primes_0, l_random_x, 1),
                l_waveform_derivative(l_primes_0, l_random_x, 2),
                l_waveform_derivative(l_primes_0, l_random_x, 3),
                l_waveform_derivative(l_primes_1, l_random_x, 0),
                l_waveform_derivative(l_primes_1, l_random_x, 1),
                l_waveform_derivative(l_primes_1, l_random_x, 2),
                l_waveform_derivative(l_primes_1, l_random_x, 3),
            });

            set_state(l_desired_y[0], l_evaluate_waveform(l_distribution, l_random_x));

            l_model.fwd();

            l_average_loss += l_loss.m_state;

            l_loss.m_partial_gradient = 1;

            l_model.bwd();

        }

        l_average_loss /= PREDICTION_COUNT;

        return l_average_loss;
        
    };

    gradient_descent_with_momentum<l_parameter_vector.size()> l_optimizer(l_parameter_vector, true, 0.2, 0.9);

    constexpr size_t MINIBATCH_SIZE = 2;

    double l_epoch_loss_running_average = 0;
    constexpr double EPOCH_LOSS_RUNNING_AVG_BETA = 0.99;

    for (int l_epoch = 0; l_epoch < 1000000000; l_epoch++)
    {
        double l_epoch_loss = 0;
        
        for (int i = 0; i < MINIBATCH_SIZE; i++)
            l_epoch_loss += l_evaluate_loss();

        l_epoch_loss /= MINIBATCH_SIZE;

        l_epoch_loss_running_average =
            EPOCH_LOSS_RUNNING_AVG_BETA * l_epoch_loss_running_average +
            (1.0 - EPOCH_LOSS_RUNNING_AVG_BETA) * l_epoch_loss;

        l_optimizer.update();

        if (l_epoch % 100 == 0)
            std::cout << "EPOCH: " << l_epoch << ", AVG. LOSS: " << l_epoch_loss_running_average << std::endl;
        
        //sleep(1);
    }

}

#pragma region DISCRETE MATH

/////////////////////////////
// Below are discrete math functions (disjoin, conjoin, invert, majority of 3, etc.)

state_gradient_pair* maj_3(
    state_gradient_pair* a_x_0,
    state_gradient_pair* a_x_1,
    state_gradient_pair* a_x_2
)
{
    auto l_term_0 = multiply(input(0.5), a_x_0);
    auto l_term_1 = multiply(input(0.5), a_x_1);
    auto l_term_2 = multiply(input(0.5), a_x_2);
    auto l_term_3 =
        multiply(
            input(-0.5), 
            multiply(
                a_x_0,
                multiply(a_x_1, a_x_2)
            )
        );

    return add(l_term_0, add(l_term_1, add(l_term_2, l_term_3)));

}

state_gradient_pair* invert(
    state_gradient_pair* a_x
)
{
    return negate(a_x);
}

state_gradient_pair* conjoin(
    state_gradient_pair* a_x_0,
    state_gradient_pair* a_x_1
)
{
    return maj_3(a_x_0, a_x_1, input(-1.0));
}

state_gradient_pair* disjoin(
    state_gradient_pair* a_x_0,
    state_gradient_pair* a_x_1
)
{
    return maj_3(a_x_0, a_x_1, input(1.0));
}

state_gradient_pair* xorb(
    state_gradient_pair* a_x_0,
    state_gradient_pair* a_x_1
)
{
    return disjoin(
        conjoin(a_x_0, invert(a_x_1)),
        conjoin(invert(a_x_0), a_x_1)
    );
}

state_gradient_pair* equiv(
    state_gradient_pair* a_x_0,
    state_gradient_pair* a_x_1
)
{
    return invert(xorb(a_x_0, a_x_1));
}

void add1b(
    state_gradient_pair*& a_s,
    state_gradient_pair*& a_cout,
    state_gradient_pair* a_x_0,
    state_gradient_pair* a_x_1,
    state_gradient_pair* a_cin = input(-1.0)
)
{
    a_s = xorb(a_cin, xorb(a_x_0, a_x_1)); // Compute the 1b-sum
    a_cout = disjoin(                      // Compute the 1b-carry
        conjoin(a_cin, a_x_0),
        disjoin(
            conjoin(a_cin, a_x_1),
            conjoin(a_x_0, a_x_1))
    );
}

template<size_t OPERAND_SIZE>
latent_tensor<OPERAND_SIZE + 1> addnb(
    const latent_tensor<OPERAND_SIZE>& a_x_0,
    const latent_tensor<OPERAND_SIZE>& a_x_1,
    state_gradient_pair* a_cin = input(-1.0)
)
{
    latent_tensor<OPERAND_SIZE + 1> l_result;
    
    // Assume LSB is at index 0.

    state_gradient_pair* l_carry = a_cin;

    for (int i = 0; i < OPERAND_SIZE; i++)
    {
        add1b(l_result[i], l_carry, a_x_0[i], a_x_1[i], l_carry);
    }

    l_result.back() = l_carry;

    return l_result;

}

template<size_t SHIFT_AMOUNT, size_t OPERAND_SIZE>
latent_tensor<OPERAND_SIZE + SHIFT_AMOUNT> pad_front(
    const latent_tensor<OPERAND_SIZE>& a_x
)
{
    latent_tensor<OPERAND_SIZE + SHIFT_AMOUNT> l_result;

    // Although confusing, the LEFT in left_shift is concerning
    // the value represented in binary, and since we represent
    // these values with LSB first, we will actually shift the values to the right.

    for (int i = 0; i < SHIFT_AMOUNT; i++)
    {
        // Initialize all of the padding values.
        l_result[i] = input(-1.0);
    }

    for (int i = 0; i < OPERAND_SIZE; i++)
    {
        // Copy the content values over.
        l_result[SHIFT_AMOUNT + i] = a_x[i];
    }

    return l_result;
    
}

template<size_t PADDING, size_t OPERAND_SIZE>
latent_tensor<OPERAND_SIZE + PADDING> pad_back(
    const latent_tensor<OPERAND_SIZE>& a_x
)
{
    latent_tensor<OPERAND_SIZE + PADDING> l_result;

    // Although confusing, the LEFT in left_shift is concerning
    // the value represented in binary, and since we represent
    // these values with LSB first, we will actually shift the values to the right.

    for (int i = 0; i < OPERAND_SIZE; i++)
    {
        // Copy the content values over.
        l_result[i] = a_x[i];
    }
    
    for (int i = 0; i < PADDING; i++)
    {
        // Initialize all of the padding values.
        l_result[OPERAND_SIZE + i] = input(-1.0);
    }

    return l_result;
    
}

template<size_t RESULT_SIZE, size_t OPERAND_SIZE>
latent_tensor<RESULT_SIZE> first(
    const latent_tensor<OPERAND_SIZE>& a_x
)
{
    latent_tensor<RESULT_SIZE> l_result;

    for (int i = 0; i < RESULT_SIZE; i++)
        l_result[i] = a_x[i];

    return l_result;
    
}

template<size_t OPERAND_SIZE, size_t CARRY_POSITION = 0>
latent_tensor<2 * OPERAND_SIZE> multiplynb(
    const latent_tensor<OPERAND_SIZE>& a_x_0,
    const latent_tensor<OPERAND_SIZE>& a_x_1
)
{
    // a_x_1 is "top" operand 
    // a_x_0 is "bottom" operand.
    
    latent_tensor<OPERAND_SIZE> l_conjunction_result;

    for (int i = 0; i < OPERAND_SIZE; i++)
        l_conjunction_result[i] = conjoin(a_x_0[CARRY_POSITION], a_x_1[i]);

    latent_tensor<2 * OPERAND_SIZE> l_padded_result =
        pad_front<CARRY_POSITION>(
            pad_back<OPERAND_SIZE - CARRY_POSITION>(l_conjunction_result)
        );

    // We now have the conjunction result, so do a recursive call then return.

    if constexpr (CARRY_POSITION == OPERAND_SIZE - 1)
    {
        return l_padded_result;
    }
    else
    {
        return 
            first<2 * OPERAND_SIZE>(
                addnb(
                    l_padded_result,
                    multiplynb<OPERAND_SIZE, CARRY_POSITION + 1>(
                        a_x_0,
                        a_x_1
                    )
                )
            );
    }

}

template<size_t OPERAND_SIZE>
state_gradient_pair* equivnb(
    const latent_tensor<OPERAND_SIZE>& a_x_0,
    const latent_tensor<OPERAND_SIZE>& a_x_1
)
{
    state_gradient_pair* l_result = input(1.0);

    for (int i = 0; i < OPERAND_SIZE; i++)
        l_result = conjoin(l_result, equiv(a_x_0[i], a_x_1[i]));

    return l_result;
    
}

void test_add1b(

)
{
    model::begin();
    
    state_gradient_pair* l_s;
    state_gradient_pair* l_cout;

    auto l_x_0 = input();
    auto l_x_1 = input();
    auto l_cin = input();

    add1b(l_s, l_cout, l_x_0, l_x_1, l_cin);

    model l_model = model::end();

    l_x_0->m_state = -1.0;
    l_x_1->m_state = -1.0;
    l_cin->m_state = -1.0;

    l_model.fwd();

    assert(l_s->m_state == -1.0);
    assert(l_cout->m_state == -1.0);

    
    l_x_0->m_state = -1.0;
    l_x_1->m_state = -1.0;
    l_cin->m_state = 1.0;

    l_model.fwd();

    assert(l_s->m_state == 1.0);
    assert(l_cout->m_state == -1.0);

    
    l_x_0->m_state = -1.0;
    l_x_1->m_state = 1.0;
    l_cin->m_state = -1.0;

    l_model.fwd();

    assert(l_s->m_state == 1.0);
    assert(l_cout->m_state == -1.0);

    
    l_x_0->m_state = -1.0;
    l_x_1->m_state = 1.0;
    l_cin->m_state = 1.0;

    l_model.fwd();

    assert(l_s->m_state == -1.0);
    assert(l_cout->m_state == 1.0);

    
    l_x_0->m_state = 1.0;
    l_x_1->m_state = -1.0;
    l_cin->m_state = -1.0;

    l_model.fwd();

    assert(l_s->m_state == 1.0);
    assert(l_cout->m_state == -1.0);

    
    l_x_0->m_state = 1.0;
    l_x_1->m_state = -1.0;
    l_cin->m_state = 1.0;

    l_model.fwd();

    assert(l_s->m_state == -1.0);
    assert(l_cout->m_state == 1.0);

    
    l_x_0->m_state = 1.0;
    l_x_1->m_state = 1.0;
    l_cin->m_state = -1.0;

    l_model.fwd();

    assert(l_s->m_state == -1.0);
    assert(l_cout->m_state == 1.0);

    
    l_x_0->m_state = 1.0;
    l_x_1->m_state = 1.0;
    l_cin->m_state = 1.0;

    l_model.fwd();

    assert(l_s->m_state == 1.0);
    assert(l_cout->m_state == 1.0);
    
}

void test_add8b(

)
{
    model::begin();

    auto l_x_0 = input<8>();
    auto l_x_1 = input<8>();
    auto l_cin = input(-1.0);

    latent_tensor<9> l_sum = addnb<8>(l_x_0, l_x_1, l_cin);

    model l_model = model::end();

    set_state(l_x_0, 
    {
        -1.0, // 0
        1.0,  // 1
        -1.0, // 2
        -1.0, // 3
        1.0,  // 4
        -1.0, // 5
        1.0,  // 6
        -1.0  // 7
    });

    set_state(l_x_1, 
    {
        -1.0, // 0
        1.0,  // 1
        1.0,  // 2
        1.0,  // 3
        1.0,  // 4
        -1.0, // 5
        1.0,  // 6
        -1.0  // 7
    });

    l_model.fwd();

    assert(l_sum[0]->m_state == -1.0);
    assert(l_sum[1]->m_state == -1.0);
    assert(l_sum[2]->m_state == -1.0);
    assert(l_sum[3]->m_state == -1.0);
    assert(l_sum[4]->m_state == 1.0);
    assert(l_sum[5]->m_state == 1.0);
    assert(l_sum[6]->m_state == -1.0);
    assert(l_sum[7]->m_state == 1.0);
    assert(l_sum[8]->m_state == -1.0);

    // Imagine setting the input carry bit.
    l_cin->m_state = 1.0;

    set_state(l_x_0, 
    {
        -1.0, // 0
        -1.0, // 1
        -1.0, // 2
        -1.0, // 3
        -1.0, // 4
        1.0,  // 5
        1.0,  // 6
        -1.0  // 7
    });

    set_state(l_x_1, 
    {
        1.0,  // 0
        -1.0, // 1
        -1.0, // 2
        -1.0, // 3
        -1.0, // 4
        1.0,  // 5
        1.0,  // 6
        1.0   // 7
    });

    l_model.fwd();

    assert(l_sum[0]->m_state == -1.0);
    assert(l_sum[1]->m_state == 1.0);
    assert(l_sum[2]->m_state == -1.0);
    assert(l_sum[3]->m_state == -1.0);
    assert(l_sum[4]->m_state == -1.0);
    assert(l_sum[5]->m_state == -1.0);
    assert(l_sum[6]->m_state == 1.0);
    assert(l_sum[7]->m_state == -1.0);
    assert(l_sum[8]->m_state == 1.0);

}

void test_left_shift(

)
{
    model::begin();

    auto l_x = input<8>();

    latent_tensor<10> l_sum = pad_front<2>(l_x);

    model l_model = model::end();

    set_state(l_x, 
    {
        -1.0, // 0
        1.0,  // 1
        -1.0, // 2
        -1.0, // 3
        1.0,  // 4
        -1.0, // 5
        1.0,  // 6
        -1.0  // 7
    });

    l_model.fwd();

    assert(l_sum[0]->m_state == -1.0); // PADDING
    assert(l_sum[1]->m_state == -1.0); // PADDING
    assert(l_sum[2]->m_state == -1.0);
    assert(l_sum[3]->m_state == 1.0);
    assert(l_sum[4]->m_state == -1.0);
    assert(l_sum[5]->m_state == -1.0);
    assert(l_sum[6]->m_state == 1.0);
    assert(l_sum[7]->m_state == -1.0);
    assert(l_sum[8]->m_state == 1.0);
    assert(l_sum[9]->m_state == -1.0);

}

void test_pad_back(

)
{
    model::begin();

    auto l_x = input<8>();

    latent_tensor<10> l_sum = pad_back<2>(l_x);

    model l_model = model::end();

    set_state(l_x, 
    {
        -1.0, // 0
        1.0,  // 1
        -1.0, // 2
        -1.0, // 3
        1.0,  // 4
        -1.0, // 5
        1.0,  // 6
        -1.0  // 7
    });

    l_model.fwd();

    assert(l_sum[0]->m_state == -1.0);
    assert(l_sum[1]->m_state == 1.0);
    assert(l_sum[2]->m_state == -1.0);
    assert(l_sum[3]->m_state == -1.0);
    assert(l_sum[4]->m_state == 1.0);
    assert(l_sum[5]->m_state == -1.0);
    assert(l_sum[6]->m_state == 1.0);
    assert(l_sum[7]->m_state == -1.0);
    assert(l_sum[8]->m_state == -1.0); // PADDING
    assert(l_sum[9]->m_state == -1.0); // PADDING

}

void test_multiplynb(

)
{
    model::begin();

    auto l_x_0 = input<4>();
    auto l_x_1 = input<4>();

    auto l_product = multiplynb(l_x_0, l_x_1);

    model l_model = model::end();

    set_state(l_x_0, 
    {
        -1.0, // 0
        1.0,  // 1
        -1.0, // 2
        -1.0, // 3
    });
    
    set_state(l_x_1, 
    {
        1.0, // 0
        1.0,  // 1
        -1.0, // 2
        1.0, // 3
    });

    l_model.fwd();

    assert(l_product[0]->m_state == -1.0);
    assert(l_product[1]->m_state == 1.0);
    assert(l_product[2]->m_state == 1.0);
    assert(l_product[3]->m_state == -1.0);
    assert(l_product[4]->m_state == 1.0);
    assert(l_product[5]->m_state == -1.0);
    assert(l_product[6]->m_state == -1.0);
    assert(l_product[7]->m_state == -1.0);
    

    // TEST ALL 1's

    set_state(l_x_0, 
    {
        1.0, // 0
        1.0,  // 1
        1.0, // 2
        1.0, // 3
    });
    
    set_state(l_x_1, 
    {
        1.0, // 0
        1.0,  // 1
        1.0, // 2
        1.0, // 3
    });

    l_model.fwd();

    assert(l_product[0]->m_state == 1.0);
    assert(l_product[1]->m_state == -1.0);
    assert(l_product[2]->m_state == -1.0);
    assert(l_product[3]->m_state == -1.0);
    assert(l_product[4]->m_state == -1.0);
    assert(l_product[5]->m_state == 1.0);
    assert(l_product[6]->m_state == 1.0);
    assert(l_product[7]->m_state == 1.0);
    

    // TEST ALL 0's

    set_state(l_x_0, 
    {
        -1.0, // 0
        -1.0,  // 1
        -1.0, // 2
        -1.0, // 3
    });
    
    set_state(l_x_1, 
    {
        -1.0, // 0
        -1.0,  // 1
        -1.0, // 2
        -1.0, // 3
    });

    l_model.fwd();

    assert(l_product[0]->m_state == -1.0);
    assert(l_product[1]->m_state == -1.0);
    assert(l_product[2]->m_state == -1.0);
    assert(l_product[3]->m_state == -1.0);
    assert(l_product[4]->m_state == -1.0);
    assert(l_product[5]->m_state == -1.0);
    assert(l_product[6]->m_state == -1.0);
    assert(l_product[7]->m_state == -1.0);
    


    // TEST ANOTHER CONTINGENCY

    set_state(l_x_0, 
    {
        1.0, // 0
        1.0,  // 1
        1.0, // 2
        -1.0, // 3
    });
    
    set_state(l_x_1, 
    {
        -1.0, // 0
        1.0,  // 1
        -1.0, // 2
        1.0, // 3
    });

    l_model.fwd();

    assert(l_product[0]->m_state == -1.0);
    assert(l_product[1]->m_state == 1.0);
    assert(l_product[2]->m_state == 1.0);
    assert(l_product[3]->m_state == -1.0);
    assert(l_product[4]->m_state == -1.0);
    assert(l_product[5]->m_state == -1.0);
    assert(l_product[6]->m_state == 1.0);
    assert(l_product[7]->m_state == -1.0);
    
}

void test_boolean_expression_modeling(

)
{
    std::uniform_real_distribution<double> l_urd(-5, 5);
    
    auto l_generate_parameter = [&l_urd]{
        return l_urd(i_random_engine);
    };
    
    model::begin();
    
    auto l_parameter_vector = input<4>(l_generate_parameter);
    auto l_boolean_vector = tanh(l_parameter_vector);

    auto a = l_boolean_vector[0];
    auto b = l_boolean_vector[1];
    auto c = l_boolean_vector[2];
    auto d = l_boolean_vector[3];
    
    // Describe the boolean expression.
    auto l_y = 
        conjoin(
            disjoin(
                invert(
                    conjoin(
                        a, b
                    )
                ),
                invert(
                    c
                )
            ),
            disjoin(
                a,
                d
            )
        );

    // We want the boolean function to be satisfied.
    auto l_desired_y = input(1.0);

    auto l_loss = squared_error(l_y, l_desired_y)->depend();

    model l_model = model::end();

    gradient_descent_with_momentum l_optimizer(
        l_parameter_vector,
        false,
        0.02,
        0.9
    );

    for (int l_epoch = 0; true; l_epoch++)
    {
        l_model.fwd();
        
        if (l_epoch % 1000 == 0)
            std::cout << "CURRENT SAT STATE: " << l_y->m_state << std::endl;
        
        l_loss.m_partial_gradient = 1;

        l_model.bwd();

        l_optimizer.update();
        
    }
    
}

void test_factoring(

)
{
    std::uniform_real_distribution<double> l_urd(-1, 1);
    
    auto l_generate_parameter = [&l_urd]{
        return l_urd(i_random_engine);
    };
    
    model::begin();

    constexpr size_t FACTOR_SIZE = 8;

    auto l_x_0_params = input<FACTOR_SIZE>(l_generate_parameter);
    auto l_x_0_params_tanh = tanh(l_x_0_params);
    
    auto l_x_1_params = input<FACTOR_SIZE>(l_generate_parameter);
    auto l_x_1_params_tanh = tanh(l_x_1_params);

    // We now have the randomly initialized factors.

    auto l_y = multiplynb(l_x_0_params_tanh, l_x_1_params_tanh);

    auto l_desired = input<2 * FACTOR_SIZE>();

    auto l_loss = invert(equivnb(l_desired, l_y))->depend();

    model l_model = model::end();

    gradient_descent_with_momentum l_optimizer(
        flatten(l_x_0_params, l_x_1_params),
        true,
        0.002,
        0.9
    );

    // Now, we select a product which we wish to factor.

    set_state(l_desired,
    {
        1.0,
        -1.0,
        -1.0,
        -1.0, //
        -1.0,
        1.0,
        -1.0,
        -1.0, //
        1.0,
        1.0,
        1.0,
        1.0,  //
        1.0,
        -1.0,
        1.0,
        1.0   //
    });

    for (int l_epoch = 0; true; l_epoch++)
    {
        l_model.fwd();
        l_loss.m_partial_gradient = 1;
        l_model.bwd();
        l_optimizer.update();
        if (l_epoch % 1000 == 0)
            std::cout << l_loss.m_state << std::endl;
    }


}

void custom_tests(

)
{
    test_add1b();
    test_add8b();
    test_left_shift();
    test_pad_back();
    test_multiplynb();
    //test_boolean_expression_modeling();
    test_factoring();
}

#pragma endregion

int main(

)
{
    //custom_tests();

    unit_test_main();

	return 0;

}
