#include "aurora-v4/aurora.h"
#include <assert.h>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <string>
#include <filesystem>

using namespace aurora;

// Returns the number of milliseconds elapsed since the start.
long long duration_ms(
    const std::chrono::high_resolution_clock::time_point& a_start
)
{
    return (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - a_start)).count();
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
	oneshot::particle_swarm_optimizer<double, PARTICLE_COUNT, PARAMETER_VECTOR_SIZE> 
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
    constexpr size_t NODE_COUNT = 100;
    constexpr size_t PARTICLE_COUNT = 50;
    constexpr size_t NUMBER_OF_EVALUATIONS_IN_SPAN = 10000;
    
    // In this scatter span, we start off with a list of current states.
    // Then, we repeatedly do the following:
    // Select a random 2-permutation (x0, x1) of the list of current states.
    // Input the 2-perm into the operation N.
    // Replace one of the input operands in the list of current states with the output of N.

    constexpr size_t WAVEFORM_SIZE = 10;
    constexpr std::array<size_t, 3> R_DIMS = { 2 * WAVEFORM_SIZE, WAVEFORM_SIZE, 1 };

    std::mt19937 l_dre(26);
    std::uniform_real_distribution<double> l_urd(-1, 1);
    std::uniform_real_distribution<double> l_waveform_urd(-10, 10);
    
    std::function<double()> l_randomly_generate_parameter = [&l_dre, &l_urd] { return l_urd(l_dre); };
    std::function<double()> l_randomly_generate_waveform_value = [&l_dre, &l_waveform_urd] { return l_waveform_urd(l_dre); };

    auto l_R_w0 =        constant<double, R_DIMS[1], R_DIMS[0]>();
    auto l_R_b1 =        constant<double, R_DIMS[1]>();
    auto l_R_w1 =        constant<double, R_DIMS[2], R_DIMS[1]>();
    auto l_R_b2 =        constant<double, R_DIMS[2]>();
    auto l_N_0 =         constant<double, WAVEFORM_SIZE, 2 * WAVEFORM_SIZE>();
    auto l_N_1 =         constant<double, WAVEFORM_SIZE, 2 * WAVEFORM_SIZE>();
    auto l_waveforms_0 = constant<double, NODE_COUNT, WAVEFORM_SIZE>();
    auto l_waveforms_1 = constant<double, NODE_COUNT, WAVEFORM_SIZE>();

    constexpr size_t PARAMETER_VECTOR_SIZE =
        l_R_w0.flattened_size() +
        l_R_b1.flattened_size() +
        l_R_w1.flattened_size() +
        l_R_b2.flattened_size() +
        l_N_0.flattened_size() +
        l_N_1.flattened_size() +
        l_waveforms_0.flattened_size() +
        l_waveforms_1.flattened_size();

    auto l_R = [&](
        const tensor<double, WAVEFORM_SIZE>& a_x_0,
        const tensor<double, WAVEFORM_SIZE>& a_x_1
    )
    {
        auto l_x = concat(a_x_0, a_x_1);
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
        copy(a_parameter_vector, l_R_w0, l_R_b1, l_R_w1, l_R_b2, l_N_0, l_N_1, l_waveforms_0, l_waveforms_1);

        // Allocate memory for waveform labels.
        tensor<double, NODE_COUNT> l_waveform_labels;

        // Use the classifier to classify roots of the span
        for (int i = 0; i < NODE_COUNT; i++)
        {
            double l_prediction = l_R(l_waveforms_0[i], l_waveforms_1[i]);
            l_waveform_labels[i] = double(l_prediction > 0.5);
        }

        // THE TRIVIAL SOLUTION IS NOT POSSIBLE HERE, SINCE
        // IF ALL WAVEFORMS ARE CONSIDERED TO BE 0's, THEN 
        // BY EVALUATING N ON THEM ONCE, IT SHOULD RETURN 1.
        // THE CONVERSE IS ALSO TRUE.

        double l_accuracy = 0;

        for (int i = 0; i < NUMBER_OF_EVALUATIONS_IN_SPAN; i++)
        {
            // We will do this a number of times equal to the
            // number of evaluations we should have in each span.
            
            size_t l_random_index_0 = rand() % NODE_COUNT;
            size_t l_random_index_1 = rand() % NODE_COUNT;

            tensor<double, WAVEFORM_SIZE> l_N_0_y = multiply(
                l_N_0,
                concat(
                    l_waveforms_0[l_random_index_0],
                    l_waveforms_0[l_random_index_1]
                )
            );

            tensor<double, WAVEFORM_SIZE> l_N_1_y = multiply(
                l_N_1,
                concat(
                    l_waveforms_1[l_random_index_0],
                    l_waveforms_1[l_random_index_1]
                )
            );

            double l_R_y = l_R(l_N_0_y, l_N_1_y);
            double l_R_y_label = 
                double(
                    !(l_waveform_labels[l_random_index_0] && l_waveform_labels[l_random_index_1]) // NAND
                );

            l_accuracy += (1 - std::abs(l_R_y - l_R_y_label));
            
            // Now that we've collected reward, go ahead and replace one of
            // the operands with the output and correct label.

            size_t l_replacement_index = (rand() % 2) ? l_random_index_0 : l_random_index_1;

            l_waveforms_0[l_replacement_index] =     l_N_0_y;
            l_waveforms_1[l_replacement_index] =     l_N_1_y;
            l_waveform_labels[l_replacement_index] = l_R_y_label;

        }

        l_accuracy /= (double)NUMBER_OF_EVALUATIONS_IN_SPAN;

        return l_accuracy;
        
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

    oneshot::particle_swarm_optimizer l_optimizer(l_positions, 0.9, 0.2, 0.8, 1.0);

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
}

int main(

)
{
    nonlinear_scatter_span_linearization();

    unit_test_main();

	return 0;

}
