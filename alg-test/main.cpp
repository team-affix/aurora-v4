#include "aurora-v4/aurora.h"
#include <assert.h>
#include <iostream>
#include <chrono>
#include <stdexcept>

using namespace aurora;
using namespace aurora::latent;

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
	element_vector::start();
	parameter_vector::start();

	auto l_x = input(4, 2);
	auto l_desired_y = input(4, 1);

	sgp_ptr_matrix l_y;

	size_t l_next_parameter_index = parameter_vector::next_index();

	for (int i = 0; i < l_x.size(); i++)
	{
		parameter_vector::next_index(l_next_parameter_index);
		auto l_y_element = pointers(l_x[i]);
		l_y_element = weight_junction(l_y_element, 5);
		l_y_element = bias(l_y_element);
		l_y_element = leaky_relu(l_y_element, 0.3);
		l_y_element = weight_junction(l_y_element, 1);
		l_y_element = bias(l_y_element);
		l_y_element = sigmoid(l_y_element);
		l_y.push_back(l_y_element);
	}
	
	sgp_ptr_vector l_cross_entropy_losses;

	for (int i = 0; i < l_y.size(); i++)
		l_cross_entropy_losses.push_back(cross_entropy(l_y[i][0], &l_desired_y[i][0]));

	auto l_error = additive_aggregate(l_cross_entropy_losses)->depend();

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(25);

	for (int i = 0; i < l_parameters.size(); i++)
	{
		l_parameters[i]->m_state = l_urd(l_dre);
	}

	const int CHECKPOINT = 100000;

	state_matrix l_ts_x =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	state_matrix l_ts_y =
	{
		{0},
		{1},
		{1},
		{0}
	};

    std::chrono::time_point l_start = std::chrono::high_resolution_clock::now();

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		set_state(pointers(l_x), l_ts_x);
		set_state(pointers(l_desired_y), l_ts_y);

		l_model.fwd();

		l_error.m_partial_gradient = 1;

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

		for (int i = 0; i < l_parameters.size(); i++)
		{
			l_parameters[i]->m_state -= 0.002 * l_parameters[i]->gradient();
		}

	}

	std::cout 
        << std::endl << "PERIOD OF TRAINING (ms): " 
        << duration_ms(l_start)
        << std::endl;

}

double sign_d(const double& a_double)
{
	if (a_double >= 0)
		return 1.0;
	else
		return -1.0;
}

void parabola_test(

)
{
	auto l_x = input(10, 1);

	element_vector::start();
	parameter_vector::start();
	
	sgp_ptr_matrix l_y(l_x.size());

	size_t l_next_parameter_index = parameter_vector::next_index();

	for (int i = 0; i < l_y.size(); i++)
	{
		parameter_vector::next_index(l_next_parameter_index);
		auto l_tnn_y = pointers(l_x[i]);
		l_tnn_y = weight_junction(l_tnn_y, 20);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
		l_tnn_y = weight_junction(l_tnn_y, 1);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
		l_y[i] = l_tnn_y;
	}

	auto l_desired_y = input(l_y.size(), l_y[0].size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y))->depend();

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	std::default_random_engine l_dre(25);

	std::uniform_real_distribution<double> l_ts_urd(-10, 10);

	double l_cost_momentum = 0;

	const size_t CHECKPOINT_INTERVAL = 10000;

	for (int epoch = 0; true; epoch++)
	{
		double l_cost = 0;

		state_matrix l_ts_x;
		state_matrix l_ts_y;

		for (int i = 0; i < l_x.size(); i++)
		{
			l_ts_x.push_back({ l_ts_urd(l_dre) });
			l_ts_y.push_back({ l_ts_x.back()[0] * l_ts_x.back()[0] });
		}

		set_state(pointers(l_x), l_ts_x);
		set_state(pointers(l_desired_y), l_ts_y);

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

		for (int i = 0; i < l_parameters.size(); i++)
		{
			l_parameters[i]->m_state -= 0.002 * tanh(l_parameters[i]->gradient());
		}

		if (epoch % CHECKPOINT_INTERVAL == 0)
			std::cout << "    LOSS FOR ABOVE EPOCH: " << l_cost_momentum << std::endl;

	}

}

void lstm_test(

)
{
	element_vector::start();
	parameter_vector::start();

	const size_t l_lstm_y_units = 3;
	const size_t l_tnn_h0_units = 3;
	const size_t l_tnn_y_units = 1;

	auto l_x = input(2, 4, 2);

	sgp_ptr_cuboid l_y;

	size_t l_model_begin_next_parameter_index = parameter_vector::next_index();

	for (int i = 0; i < l_x.size(); i++)
	{
		parameter_vector::next_index(l_model_begin_next_parameter_index);
		auto l_lstm_y = lstm(pointers(l_x[i]), l_lstm_y_units);
		size_t l_tnn_begin_next_parameter_index = parameter_vector::next_index();
		sgp_ptr_matrix l_tnn_ys;
		for (int j = 0; j < l_lstm_y.size(); j++)
		{
			parameter_vector::next_index(l_tnn_begin_next_parameter_index);
			auto l_tnn_y = l_lstm_y[j];
			l_tnn_y = weight_junction(l_tnn_y, l_tnn_h0_units);
			l_tnn_y = bias(l_tnn_y);
			l_tnn_y = leaky_relu(l_tnn_y, 0.3);
			l_tnn_y = weight_junction(l_tnn_y, l_tnn_y_units);
			l_tnn_y = bias(l_tnn_y);
			l_tnn_y = leaky_relu(l_tnn_y, 0.3);
			l_tnn_ys.push_back(l_tnn_y);
		}
		l_y.push_back(l_tnn_ys);
	}

	auto l_desired_y = input(l_y.size(), l_y[0].size(), l_y[0][0].size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y))->depend();

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, true, 0.02);

	state_cuboid l_training_set_xs =
	{
		{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		},
		{
			{0, 1},
			{0, 1},
			{1, 0},
			{1, 1}
		},
	};

	state_cuboid l_training_set_ys =
	{
		{
			{0},
			{1},
			{1},
			{0}
		},
		{
			{0},
			{1},
			{1},
			{1}
		},
	};

	const size_t CHECKPOINT = 10000;

	for (int epoch = 0; true; epoch++)
	{
		double l_cost = 0;

		set_state(pointers(l_x), l_training_set_xs);
		set_state(pointers(l_desired_y), l_training_set_ys);

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
				std::cout << "PREDICTION: \n" << to_string(get_state(l_y)) << std::endl;
			std::cout << std::endl;
		}

		l_optimizer.update();

		if (epoch % CHECKPOINT == 0)
			std::cout << "COST: " << l_cost << std::endl << std::endl;

	}

}

void lstm_stacked_test(

)
{
	element_vector::start();
	parameter_vector::start();

	auto l_x = input(4, 2);

	auto l_lstm_0 = lstm(pointers(l_x), 20);
	auto l_lstm_1 = lstm(l_lstm_0, 20);
	auto l_lstm_2 = lstm(l_lstm_1, 1);

	sgp_ptr_matrix l_y;

	for (int i = 0; i < l_lstm_2.size(); i++)
	{
		auto l_tnn_y = l_lstm_2[i];
		l_tnn_y = weight_junction(l_tnn_y, 5);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
		l_tnn_y = weight_junction(l_tnn_y, 1);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
		l_y.push_back(l_tnn_y);
	}

	auto l_desired_y = input(l_y.size(), l_y[0].size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y))->depend();

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(28);

	for (auto& l_parameter : l_parameters)
	{
		l_parameter->m_state = l_urd(l_dre);
	}

	state_cuboid l_training_set_xs =
	{
		{
			{0, 0},
			{0, 1},
			{1, 0},
			{1, 1}
		},
		{
			{0, 1},
			{0, 1},
			{1, 0},
			{1, 1}
		},
	};

	state_cuboid l_training_set_ys =
	{
		{
			{0},
			{1},
			{1},
			{0}
		},
		{
			{0},
			{1},
			{1},
			{1}
		},
	};

	size_t CHECKPOINT = 100;

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		double l_cost = 0;

		for (int i = 0; i < l_training_set_xs.size(); i++)
		{
			set_state(pointers(l_x), l_training_set_xs[i]);
			set_state(pointers(l_desired_y), l_training_set_ys[i]);

			l_model.fwd();
			l_cost += l_error.m_state;
			l_error.m_partial_gradient = 1;
			l_model.bwd();
		}

		for (auto& l_parameter : l_parameters)
		{
			l_parameter->m_state -= 0.2 * l_parameter->gradient();
		}

		if (epoch % CHECKPOINT == 0)
			std::cout << l_cost << std::endl;

	}

}

void matrix_vector_multiply_test(

)
{
	element_vector::start();

	sgp_matrix l_x_0
	{
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8}
	};

	sgp_vector l_x_1
	{
		2,
		3
	};

	auto l_y = multiply(pointers(l_x_0), pointers(l_x_1));

	element_vector l_model = element_vector::stop();

	l_model.fwd();

}

void cosine_similarity_test(

)
{
	element_vector::start();

	sgp_vector l_x_0{ 0, 1, 0, 0 };
	sgp_vector l_x_1{ 0, -1, 0, 0 };

	auto l_y = cosine_similarity(pointers(l_x_0), pointers(l_x_1));

	auto l_model = element_vector::stop();

	l_model.fwd();

}

void similarity_interpolate_test(

)
{
	sgp_matrix l_tsx =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{1, 0.75}
	};

	sgp_matrix l_tsy =
	{
		{0},
		{1},
		{1},
		{0},
		{0.862}
	};

	auto l_query = input(2);
	
	element_vector::start();

	auto l_y = similarity_interpolate(pointers(l_query), pointers(l_tsx), pointers(l_tsy));

	auto l_model = element_vector::stop();

	while (true)
	{
		std::cout << "INPUT TWO VALUES." << std::endl;
		std::cin >> l_query[0].m_state;
		std::cin >> l_query[1].m_state;
		l_model.fwd();
		std::cout << l_y[0]->m_state << std::endl;
	}


}

void large_memory_usage_test(

)
{
	element_vector::start();
	parameter_vector::start();

	sgp_vector l_x(1000);

    std::chrono::time_point l_start = std::chrono::high_resolution_clock::now();

	{
		auto l_y = pointers(l_x);
		l_y = weight_junction(l_y, 1000);
		l_y = weight_junction(l_y, 1000);
		l_y = weight_junction(l_y, 1000);

		element_vector l_model = element_vector::stop();
		auto l_parameters = parameter_vector::stop();

		std::cout << "MODEL CREATED: " << l_model.size() << " elements; " << duration_ms(l_start) << " ms" << std::endl;
        l_start = std::chrono::high_resolution_clock::now();

		std::uniform_real_distribution<double> l_urd(-1, 1);
		std::default_random_engine l_dre(25);

		for (auto& l_parameter : l_parameters)
			l_parameter->m_state = l_urd(l_dre);
		std::cout << "PARAMETERS INITIALIZED: " << duration_ms(l_start) << " ms" << std::endl;
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

sgp_ptr_cuboid in_sequence_stock_predict(
	sgp_ptr_matrix a_x,
	const std::vector<size_t>& a_lstm_y_sizes,
	const std::vector<size_t>& a_layer_y_sizes,
	const size_t& a_time_slots_to_predict_per_timestep,
	const size_t& a_time_slot_prediction_size
)
{
	sgp_ptr_matrix l_y_raw = a_x;

	for (int i = 0; i < a_lstm_y_sizes.size(); i++)
		l_y_raw = lstm(l_y_raw, a_lstm_y_sizes[i]);

	const size_t TOTAL_OUTPUT_UNITS = a_time_slot_prediction_size * a_time_slots_to_predict_per_timestep;

	for (int i = 0; i < l_y_raw.size(); i++)
	{
		for (int j = 0; j < a_layer_y_sizes.size(); j++)
		{
			l_y_raw[i] = weight_junction(l_y_raw[i], a_layer_y_sizes[j]);
			l_y_raw[i] = bias(l_y_raw[i]);
			l_y_raw[i] = leaky_relu(l_y_raw[i], 0.3);
		}
	}

	sgp_ptr_cuboid l_future_hour_predictions;

	for (int i = 0; i < l_y_raw.size(); i++)
	{
		// Link the raw output with specific time slots
		l_future_hour_predictions.push_back(partition(l_y_raw[i], a_time_slot_prediction_size));
	}

	return l_future_hour_predictions;

}

void issp_test(

)
{
	element_vector::start();
	parameter_vector::start();

	sgp_matrix l_x = input(100, 4);

	auto l_y = in_sequence_stock_predict(
		pointers(l_x),
		{ 20, 20 },
		{ 20, 40 },
		10,
		2
	);

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	l_model.fwd();

	l_model.bwd();

}

void pablo_tnn_example(

)
{
	element_vector::start();
	parameter_vector::start();

	// Write model building code here
	sgp_vector l_x = { 0, 0 };

	sgp_ptr_vector l_y = pointers(l_x);

	l_y = weight_junction(l_y, 5);
	l_y = bias(l_y);
	l_y = tanh(l_y);
	
	l_y = weight_junction(l_y, 1);
	l_y = bias(l_y);
	l_y = sigmoid(l_y);


	auto l_desired_y = input(l_y.size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y))->depend();

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, true, 0.02);

	state_matrix l_tsx =
	{
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
	};

	state_matrix l_tsy =
	{
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 },
	};

	const size_t CHECKPOINT = 100000;

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		for (int i = 0; i < l_tsx.size(); i++)
		{
			set_state(pointers(l_x), l_tsx[i]);
			set_state(pointers(l_desired_y), l_tsy[i]);

			l_model.fwd();
			l_error.m_partial_gradient = 1;
			l_model.bwd();

			if (epoch % CHECKPOINT == 0)
			{
				std::cout << l_y[0]->m_state << std::endl;
			}

		}

		l_optimizer.update();

		if (epoch % CHECKPOINT == 0)
		{
			std::cout << std::endl;
		}

	}

}

void reward_structure_modeling(

)
{
	sgp_vector l_x(3);

	// INPUTS:
	// PERCEIVED CHEAPNESS OF THE ITEM
	// PREDICTED INCREASE IN UTILITY
	// PREDICTED INCREASE IN ENJOYMENT

	// OUTPUTS:
	// DESIRE TO PURCHASE

	element_vector::start();
	parameter_vector::start();

	auto l_normalized_parameters = normalize(sigmoid(parameters(l_x.size())));

	auto l_y = multiply(l_normalized_parameters, pointers(l_x));

	auto l_desired_y = state_gradient_pair();
	auto l_error = mean_squared_error(l_y, &l_desired_y)->depend();

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, true, 0.02);

	struct training_set
	{
		state_vector m_x;
		double m_y;
	};

	std::vector<training_set> l_training_sets
	{
		training_set
		{
			{ -1, 0.1, 0.5 },
			0.25
		},
		training_set
		{
			{ 0.7, 0.05, 0 },
			0.3
		},
		training_set
		{
			{ 0.6, 0.05, -0.1 },
			-0.3
		},

	};

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		double l_cost = 0;

		for (auto& l_training_set : l_training_sets)
		{
			set_state(pointers(l_x), l_training_set.m_x);
			l_desired_y.m_state = l_training_set.m_y;

			l_model.fwd();

			l_cost += l_error.m_state;
			l_error.m_partial_gradient = 1;

			l_model.bwd();

		}

		l_optimizer.update();

		if (epoch % 10000 == 0)
			std::cout << l_cost << std::endl;

	}

	std::cout << std::endl;
	
	for (auto& l_parameter : l_normalized_parameters)
		std::cout << l_parameter->m_state << std::endl;

}

void loss_modeling_test_0(

)
{
	sgp_vector l_task_x(10);
	sgp_vector l_task_prediction(1);
	sgp_vector l_loss_model_desired_y(1);

	element_vector::start();
	parameter_vector::start();

	std::vector<size_t> l_tnn_layer_sizes = { 20, 20 };

	auto l_loss_model_y = concat(pointers(l_task_x), pointers(l_task_prediction));

	for (int i = 0; i < l_tnn_layer_sizes.size(); i++)
	{
		l_loss_model_y = weight_junction(l_loss_model_y, l_tnn_layer_sizes[i]);
		l_loss_model_y = bias(l_loss_model_y);
		l_loss_model_y = leaky_relu(l_loss_model_y, 0.3);
	}

	l_loss_model_y = weight_junction(l_loss_model_y, 1);
	l_loss_model_y = bias(l_loss_model_y);
	l_loss_model_y = leaky_relu(l_loss_model_y, 0.3);
	l_loss_model_y = { pow(l_loss_model_y[0], constant(2)) };

	auto l_loss_model_loss = mean_squared_error(l_loss_model_y, pointers(l_loss_model_desired_y))->depend();

	auto l_loss_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, true, 0.02);

	std::uniform_real_distribution<double> l_urd(-10, 10);
	std::default_random_engine l_dre(28);

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		double l_loss_model_epoch_loss = 0;

		for (int i = 0; i < 1; i++)
		{
			double l_task_desired_y = 0;

			for (int j = 0; j < l_task_x.size(); j++)
			{
				// GENERATE RANDOM TASK INPUT AND APPLY THE ACTUAL TASK TO THE INPUT (ADDITIVE ACCUMULATION IN THIS CASE)
				l_task_x[j].m_state = l_urd(l_dre);
				l_task_desired_y += l_task_x[j].m_state;
			}

			// GENERATE A RANDOM TASK PREDICTION GIVEN THIS INPUT (THIS WILL MOST LIKELY BE WRONG, BUT TO VARYING DEGREES)
			l_task_prediction[0].m_state = l_urd(l_dre);

			// CALCULATE MEAN SQUARED ERROR OF THE TASK PREDICTION
			l_loss_model_desired_y[0].m_state = std::pow(l_task_prediction[0].m_state - l_task_desired_y, 2);

			l_loss_model.fwd();
			
			l_loss_model_loss.m_partial_gradient = 1;
			l_loss_model_epoch_loss += l_loss_model_loss.m_state;

			l_loss_model.bwd();

		}

		l_optimizer.update();

		if (epoch % 10000 == 0)
		{
			std::cout << "LR: " << l_optimizer.m_learn_rate << ", LOSS: " << l_loss_model_epoch_loss << std::endl;
			l_optimizer.m_learn_rate *= 0.99;
		}

	}

	std::cout << std::endl << std::endl << "GENERATING TASK PREDICTION: " << std::endl;

	gradient_descent l_task_prediction_optimizer(pointers(l_task_prediction), true, 0.2);
	 
	double l_task_desired_y = 0;

	for (int j = 0; j < l_task_x.size(); j++)
	{
		// GENERATE RANDOM TASK INPUT AND APPLY THE ACTUAL TASK TO THE INPUT (ADDITIVE ACCUMULATION IN THIS CASE)
		l_task_x[j].m_state = l_urd(l_dre);
		l_task_desired_y += l_task_x[j].m_state;
	}

	l_loss_model_desired_y[0].m_state = 0;

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		l_loss_model.fwd();
		l_loss_model_loss.m_partial_gradient = 1;
		l_loss_model.bwd();
		l_task_prediction_optimizer.update();
		if (epoch % 10000 == 0)
		{
			std::cout << 
				"PREDICTED TASK Y: " <<
				l_task_prediction[0].m_state <<
				", DESIRED TASK Y: " <<
				l_task_desired_y <<
				std::endl;
		}
	}

	std::cout << std::endl << std::endl << "GENERATING TASK X: " << std::endl;

	gradient_descent l_task_x_optimizer(pointers(l_task_x), true, 0.2);

	l_task_prediction[0].m_state = 10000;
	l_loss_model_desired_y[0].m_state = 0;

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		l_loss_model.fwd();
		l_loss_model_loss.m_partial_gradient = 1;
		l_loss_model.bwd();
		l_task_x_optimizer.update();
		if (epoch % 10000 == 0)
		{
			std::cout <<
				"TRYING TO ACHIEVE TASK PREDICTION OF: " <<
				l_task_prediction[0].m_state <<
				", TASK X: ";
			
			for (auto& l_value : l_task_x)
				std::cout << l_value.m_state << " ";

			double l_x_sum = 0;

			for (auto& l_value : l_task_x)
				l_x_sum += l_value.m_state;

			std::cout << "WHICH YIELDS A SUM OF: " << l_x_sum << std::endl;

		}
	}


}

void tnn_test_2(

)
{
	state_matrix l_tsx =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	state_matrix l_tsy =
	{
		{0},
		{1},
		{1},
		{0},
	};
	
	element_vector::start();
	parameter_vector::start();

	auto l_x = input(2);
	auto l_y = pointers(l_x);

	std::vector<size_t> l_layer_sizes = { 5, 1 };

	for (int i = 0; i < l_layer_sizes.size(); i++)
	{
		l_y = weight_junction(l_y, l_layer_sizes[i]);
		l_y = bias(l_y);
		l_y = leaky_relu(l_y, 0.3);
	}

	auto l_desired = input(1);
	auto l_loss = mean_squared_error(l_y, pointers(l_desired))->depend();

	element_vector l_elements = element_vector::stop();
	parameter_vector l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, true, 0.02);

	const int CHECKPOINT = 100000;

	for (int epoch = 0; true; epoch++)
	{
		for (int i = 0; i < l_tsx.size(); i++)
		{
			set_state(pointers(l_x), l_tsx[i]);
			set_state(pointers(l_desired), l_tsy[i]);
			l_elements.fwd();
			l_loss.m_partial_gradient = 1;
			l_elements.bwd();
			if (epoch % CHECKPOINT == 0)
			{
				std::cout << l_y[0]->m_state << std::endl;
			}
		}
		l_optimizer.update();
		if (epoch % CHECKPOINT == 0)
		{
			std::cout << std::endl;
		}
	}

}

void convolve_test(

)
{
	element_vector::start();

	auto l_x = input(3, 50, 50);
	auto l_filter = input(3, 5, 5);
	auto l_y = convolve(pointers(l_x), pointers(l_filter), 10);

	element_vector l_element_vector = element_vector::stop();

	for (int i = 0; i < l_x.size(); i++)
	{
		for (int j = 0; j < l_x[0].size(); j++)
		{
			for (int k = 0; k < l_x[0][0].size(); k++)
			{
				l_x[i][j][k] = (i + j + k) % 100;
			}
		}
	}

	for (int i = 0; i < l_filter.size(); i++)
	{
		for (int j = 0; j < l_filter[0].size(); j++)
		{
			for (int k = 0; k < l_filter[0][0].size(); k++)
			{
				l_filter[i][j][k] = (i + j + k) % 100;
			}
		}
	}

	l_element_vector.fwd();

}

void cnn_test(

)
{
	element_vector::start();
	parameter_vector::start();

	auto l_x = input(3, 1080, 1920);
	auto l_cnn_y = convolve(pointers(l_x), parameters(3, 100, 100), 100);
	l_cnn_y = average_pool(l_cnn_y, 3, 3, 3);
	l_cnn_y = leaky_relu(l_cnn_y, 0.3);
	l_cnn_y = convolve({ l_cnn_y }, parameters(1, 2, 2));

	std::vector<size_t> l_layer_sizes = { 15, 2 };

	auto l_tnn_y = flatten(l_cnn_y);
	
	for (auto& l_layer_size : l_layer_sizes)
	{
		l_tnn_y = weight_junction(l_tnn_y, l_layer_size);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
	}

	element_vector l_element_vector = element_vector::stop();
	parameter_vector l_parameter_vector = parameter_vector::stop();



}

void oneshot_matrix_multiply_test(

)
{
	state_matrix l_matrix = 
	{
		{2, 3, 4},
		{5, 6, 7}
	};

	state_vector l_weights = { 3, 4, 5 };

	state_vector l_result = oneshot::multiply(l_matrix, l_weights);

}

void oneshot_tnn_test(

)
{
	state_matrix l_x = 
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	state_matrix l_y =
	{
		{0},
		{1},
		{1},
		{0}
	};

	oneshot::parameter_vector_builder l_parameter_vector_builder(-1, 1);
	oneshot::parameter_vector& l_parameter_vector(l_parameter_vector_builder);

	auto l_carry_forward = 
		[&l_parameter_vector, &l_x]
	{
		state_matrix l_result = l_x;

		for (int i = 0; i < l_x.size(); i++)
		{
			l_parameter_vector.next_index(0);
			l_result[i] = oneshot::multiply(l_parameter_vector.next(200, l_result[i].size()), l_result[i]);
			l_result[i] = oneshot::add(l_result[i], l_parameter_vector.next(200));
			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
			l_result[i] = oneshot::multiply(l_parameter_vector.next(1, l_result[i].size()), l_result[i]);
			l_result[i] = oneshot::add(l_result[i], l_parameter_vector.next(1));
			l_result[i] = oneshot::sigmoid(l_result[i]);
		}

		return l_result;

	};

	auto l_dry_fire = l_carry_forward();

	state_vector l_gradients(l_parameter_vector.size());

	std::uniform_real_distribution<double> l_urd(-1, 1);

	for (int i = 0; i < l_gradients.size(); i++)
		l_gradients[i] = l_urd(i_default_random_engine);

	double l_previous_reward = 1.0;

	double l_beta = 0;
	double l_learn_rate = 0.0002;

	std::uniform_real_distribution<double> l_rcv(-0.001, 0.001);

	double l_performance_momentum = 1.0;

	for (int epoch = 0; true; epoch++)
	{
		state_vector l_updates = oneshot::normalize(l_gradients);

		for (int i = 0; i < l_gradients.size(); i++)
		{
			l_updates[i] *= l_learn_rate;
			l_updates[i] += l_learn_rate / (std::abs(l_performance_momentum) + l_learn_rate) * l_rcv(i_default_random_engine);
			l_parameter_vector[i] += l_updates[i];
		}

		double l_current_reward = 1.0 / oneshot::mean_squared_error(l_carry_forward(), l_y);
		double l_change_in_reward = l_current_reward - l_previous_reward;

		l_performance_momentum =
			l_beta * l_performance_momentum +
			(1.0 - l_beta) * l_change_in_reward;

		l_previous_reward = l_current_reward;
		
		for (int i = 0; i < l_gradients.size(); i++)
		{
			double l_instantaneous_gradient_approximation = l_change_in_reward * l_updates[i] / std::pow(oneshot::magnitude(l_updates), 2.0);
			l_gradients[i] = l_beta * l_gradients[i] + (1.0 - l_beta) * l_instantaneous_gradient_approximation;
		}

		if (epoch % 100 == 0)
			std::cout << l_current_reward << std::endl;

	}

}

//void oneshot_tnn_acceleration_test(
//
//)
//{
//	state_matrix l_x =
//	{
//		{0, 0},
//		{0, 1},
//		{1, 0},
//		{1, 1}
//	};
//
//	state_matrix l_y =
//	{
//		{0},
//		{1},
//		{1},
//		{0}
//	};
//
//	oneshot::parameter_vector l_position(-1, 1);
//
//	auto l_carry_forward =
//		[&l_position, &l_x]
//	{
//		state_matrix l_result = l_x;
//
//		for (int i = 0; i < l_x.size(); i++)
//		{
//			l_position.next_index(0);
//			l_result[i] = oneshot::multiply(l_position.next(100, l_result[i].size()), l_result[i]);
//			l_result[i] = oneshot::add(l_result[i], l_position.next(100));
//			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
//			l_result[i] = oneshot::multiply(l_position.next(1, l_result[i].size()), l_result[i]);
//			l_result[i] = oneshot::add(l_result[i], l_position.next(1));
//			l_result[i] = oneshot::sigmoid(l_result[i]);
//		}
//
//		return l_result;
//
//	};
//
//	auto l_dry_fire = l_carry_forward();
//
//	std::uniform_real_distribution<double> l_velocity_urd(-0.00001, 0.00001);
//	state_vector l_velocity(l_position.size());
//	for (int i = 0; i < l_velocity.size(); i++)
//		l_velocity[i] = l_velocity_urd(i_default_random_engine);
//
//	std::uniform_real_distribution<double> l_acceleration_urd(-0.0001, 0.0001);
//	state_vector l_acceleration(l_position.size());
//	for (int i = 0; i < l_acceleration.size(); i++)
//		l_acceleration[i] = l_acceleration_urd(i_default_random_engine);
//
//	std::uniform_real_distribution<double> l_random_velocity_change(-0.001, 0.001);
//
//	double l_beta = 0.99;
//	double l_alpha = 0.002;
//
//	double l_previous_reward = 0;
//	double l_previous_change_in_reward = 0;
//
//	for (int l_epoch = 0; true; l_epoch++)
//	{
//		state_vector l_velocity_update = oneshot::multiply(oneshot::normalize(l_acceleration), l_alpha);
//		/*for (int i = 0; i < l_velocity_update.size(); i++)
//			l_velocity_update[i] += l_alpha * l_random_velocity_change(i_default_random_engine);*/
//		l_velocity = oneshot::add(l_velocity, l_velocity_update);
//		l_position = oneshot::add(l_position, l_velocity);
//		double l_reward = 1.0 / oneshot::mean_squared_error(l_carry_forward(), l_y);
//		double l_change_in_reward = l_reward - l_previous_reward;
//		double l_change_in_change_in_reward = l_change_in_reward - l_previous_change_in_reward;
//		l_previous_reward = l_reward;
//		l_previous_change_in_reward = l_change_in_reward;
//		double l_velocity_update_magnitude_squared = std::pow(oneshot::magnitude(l_velocity_update), 2);
//		for (int i = 0; i < l_acceleration.size(); i++)
//		{
//			double l_instant_acceleration = l_change_in_change_in_reward * l_velocity_update[i] / l_velocity_update_magnitude_squared;
//			// Construct a running average for the acceleration
//			l_acceleration[i] = l_beta * l_acceleration[i] + (1 - l_beta) * l_instant_acceleration;
//		}
//		std::cout << l_reward << std::endl;
//		Sleep(100);
//	}
//
//
//}

void particle_swarm_optimization_example(

)
{
	state_matrix l_x =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	state_matrix l_y =
	{
		{0},
		{1},
		{1},
		{0}
	};

	auto l_carry_forward =
		[&l_x](oneshot::parameter_vector& a_parmeter_vector)
	{
		state_matrix l_result = l_x;

		for (int i = 0; i < l_x.size(); i++)
		{
			a_parmeter_vector.next_index(0);
			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1000, l_result[i].size()), l_result[i]);
			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1000));
			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1, l_result[i].size()), l_result[i]);
			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1));
			l_result[i] = oneshot::sigmoid(l_result[i]);
		}

		return l_result;

	};

	// Initialize particle positions
	std::vector<oneshot::parameter_vector> l_particle_positions;
	for (int i = 0; i < 100; i++)
	{
		oneshot::parameter_vector_builder l_builder(-1, 1);
		l_carry_forward(l_builder); // Dry fire the particle's parameter vector
		l_particle_positions.push_back(l_builder);
	}

	// Initialize particle velocities
	state_matrix l_particle_velocities = oneshot::make(
		l_particle_positions.size(),
		l_particle_positions[0].size()
	);

	// Define hyperparameters
	double l_w = 0.9;
	double l_c1 = 0.4;
	double l_c2 = 0.6;

	state_matrix l_p_best = oneshot::make(l_particle_positions.size(), l_particle_positions[0].size());
	state_vector l_p_best_losses(l_particle_positions.size());
	for (int i = 0; i < l_p_best_losses.size(); i++)
		l_p_best_losses[i] = 9999999999999999;

	state_vector l_g_best(l_particle_positions[0].size());
	double l_g_best_loss = 9999999999999999;

	// Train
	for (int epoch = 0; true; epoch++)
	{
		// Evaluate the loss at each particle's position
		for (int i = 0; i < l_particle_positions.size(); i++)
		{
			double l_loss = oneshot::mean_squared_error(l_carry_forward(l_particle_positions[i]), l_y);
			if (l_loss < l_p_best_losses[i])
			{
				l_p_best[i] = l_particle_positions[i];
				l_p_best_losses[i] = l_loss;
			}
			if (l_loss < l_g_best_loss)
			{
				l_g_best = l_particle_positions[i];
				l_g_best_loss = l_loss;
			}
		}

		// Update the velocities of all particles
		for (int i = 0; i < l_particle_positions.size(); i++)
		{
			state_vector l_weighted_particle_velocity = oneshot::multiply(l_particle_velocities[i], l_w);
			state_vector l_cognitive_term = oneshot::multiply(oneshot::multiply(oneshot::subtract(l_p_best[i], l_particle_positions[i]), l_c1), oneshot::random(0, 1));
			state_vector l_social_term = oneshot::multiply(oneshot::multiply(oneshot::subtract(l_g_best, l_particle_positions[i]), l_c2), oneshot::random(0, 1));
			l_particle_velocities[i] = oneshot::add(oneshot::add(l_weighted_particle_velocity, l_cognitive_term), l_social_term);
		}

		// Update the positions of all particles
		for (int i = 0; i < l_particle_positions.size(); i++)
		{
			l_particle_positions[i] = oneshot::add(l_particle_positions[i], l_particle_velocities[i]);
		}

		std::cout << l_g_best_loss << std::endl;

	}

}

void particle_swarm_optimization_class_example(

)
{
	state_matrix l_x =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	state_matrix l_y =
	{
		{0},
		{1},
		{1},
		{0}
	};

	auto l_carry_forward =
		[&l_x](oneshot::parameter_vector& a_parmeter_vector)
	{
		state_matrix l_result = l_x;

		for (int i = 0; i < l_x.size(); i++)
		{
			a_parmeter_vector.next_index(0);
			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1000, l_result[i].size()), l_result[i]);
			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1000));
			l_result[i] = oneshot::leaky_relu(l_result[i], 0.3);
			l_result[i] = oneshot::multiply(a_parmeter_vector.next(1, l_result[i].size()), l_result[i]);
			l_result[i] = oneshot::add(l_result[i], a_parmeter_vector.next(1));
			l_result[i] = oneshot::sigmoid(l_result[i]);
		}

		return l_result;

	};

	// Initialize particle positions
	std::vector<oneshot::parameter_vector> l_particle_positions;

	for (int i = 0; i < 100; i++)
	{
		oneshot::parameter_vector_builder l_builder(-1, 1);
		l_carry_forward(l_builder); // Dry fire the particle's parameter vector
		l_particle_positions.push_back(l_builder);
	}

	// Define hyperparameters
	double l_w = 0.9;
	double l_c1 = 0.2;
	double l_c2 = 0.8;

	std::vector<oneshot::particle_optimizer> l_particles;
	for (int i = 0; i < l_particle_positions.size(); i++)
		l_particles.push_back(oneshot::particle_optimizer(l_particle_positions[i]));

	oneshot::particle_swarm_optimizer l_swarm_optimizer(l_particles, l_w, l_c1, l_c2);

	state_vector l_particle_rewards(l_particles.size());

	// Train
	for (int epoch = 0; true; epoch++)
	{
		for (int i = 0; i < l_particles.size(); i++)
			l_particle_rewards[i] = 1.0 / (oneshot::mean_squared_error(l_carry_forward(l_particle_positions[i]), l_y) + 0.00001);
		l_swarm_optimizer.update(l_particle_rewards);
		std::cout << l_swarm_optimizer.global_best_reward() << std::endl;
	}

}

void oneshot_partition_test(

)
{
	auto l_tensor = oneshot::random(10, 10, 10, 0, 1);
	auto l_flattened_0 = oneshot::flatten(l_tensor);
	auto l_tensor_recovered = oneshot::partition(l_flattened_0, l_tensor.size(), l_tensor[0].size(), l_tensor[0][0].size());

	assert(l_tensor_recovered == l_tensor);

	auto l_matrix = oneshot::random(10, 10, 0, 1);
	auto l_flattened_1 = oneshot::flatten(l_matrix);
	auto l_matrix_recovered = oneshot::partition(l_flattened_1, l_tensor.size(), l_tensor[0].size());

	assert(l_matrix_recovered == l_matrix);

}

void scalar_multiplication_modeling_using_matrices(

)
{
	element_vector::start();
	parameter_vector::start();

	auto l_x = input(4);

	auto l_m = bias(weight_junction(pointers(l_x), 20));
	l_m = bias(weight_junction(l_m, 20));
	auto l_y = bias(weight_junction(l_m, 1));

	auto l_desired = input(1);
	auto l_loss = mean_squared_error(l_y, pointers(l_desired))->depend();

	parameter_vector l_parameter_vector = parameter_vector::stop(-1, 1);
	element_vector l_element_vector = element_vector::stop();

	gradient_descent_with_momentum l_optimizer(l_parameter_vector, true, 0.2, 0.9);

	state_matrix l_ts_x =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};

	state_matrix l_ts_y =
	{
		{0},
		{1},
		{1},
		{1}
	};

	for (int epoch = 0; true; epoch++)
	{
		double l_epoch_cost = 0;
		for (int i = 0; i < l_ts_x.size(); i++)
		{
			set_state(pointers(l_x), l_ts_x[i]);
			set_state(pointers(l_desired), l_ts_y[i]);
			l_element_vector.fwd();
			l_loss.m_partial_gradient = 1;
			l_element_vector.bwd();
			l_epoch_cost += l_loss.m_state;
		}
		l_optimizer.update();

		if (l_epoch_cost <= 0.3)
			printf("123");

		if (epoch % 10000 == 0)
			std::cout << l_epoch_cost << std::endl;
	}

}

void test_pso(

)
{
	state_matrix l_tsx =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	state_matrix l_tsy =
	{
		{0},
		{1},
		{1},
		{0}
	};

	auto l_get_reward = [](
		aurora::oneshot::parameter_vector& a_parameter_vector,
		const state_matrix& a_x,
		const state_matrix& a_y
	)
	{
		state_matrix l_y(a_x);
		for (int i = 0; i < a_x.size(); i++)
		{
			a_parameter_vector.next_index(0);
			l_y[i] = oneshot::multiply(a_parameter_vector.next(5, 2), l_y[i]);
			l_y[i] = oneshot::add(l_y[i], a_parameter_vector.next(5));
			l_y[i] = oneshot::tanh(l_y[i]);
			l_y[i] = oneshot::multiply(a_parameter_vector.next(1, 5), l_y[i]);
			l_y[i] = oneshot::add(l_y[i], a_parameter_vector.next(1));
			l_y[i] = oneshot::tanh(l_y[i]);
		}
		return 1.0 / (oneshot::mean_squared_error(l_y, a_y) + 1E-10);
	};

	// Vector of particle independent variable positions
	std::vector<aurora::oneshot::parameter_vector> l_parameter_vectors;
	std::vector<aurora::oneshot::particle_optimizer> l_particle_optimizers;

	// Initialize the particle positions
	for (int i = 0; i < 50; i++)
	{
		oneshot::parameter_vector_builder l_builder(-7, 7);
		l_get_reward(l_builder, l_tsx, l_tsy);
		l_parameter_vectors.push_back(l_builder);
	}

	for (int i = 0; i < l_parameter_vectors.size(); i++)
		l_particle_optimizers.push_back(aurora::oneshot::particle_optimizer(l_parameter_vectors[i]));

	// Initialize the swarm optimizer.
	aurora::oneshot::particle_swarm_optimizer l_particle_swarm_optimizer(l_particle_optimizers, 0.9, 0.2, 0.8);

	// Construct a vector of the rewards associated with each parameter vector.
	state_vector l_rewards(l_parameter_vectors.size());

	for (int l_epoch = 0; true; l_epoch++)
	{
		for (int i = 0; i < l_parameter_vectors.size(); i++)
		{
			l_rewards[i] = l_get_reward(
				l_parameter_vectors[i],
				l_tsx,
				l_tsy
			);
		}
		l_particle_swarm_optimizer.update(l_rewards);
		if (l_epoch % 10 == 0)
			std::cout << l_particle_swarm_optimizer.global_best_reward() << std::endl;
	}

}

void oneshot_convolve_test(

)
{
	auto l_x = oneshot::random(3, 100, 100, 0, 1);
	auto l_filter = oneshot::random(3, 10, 10, 0, 1);
	auto l_convolved = oneshot::convolve(l_x, l_filter, 1);
	auto l_convolved_first_element =
		oneshot::multiply(
			oneshot::flatten(oneshot::range(l_x, 0, 0, 0, 3, 10, 10)),
			oneshot::flatten(l_filter));
}

void sife_concurrent_feature_extraction_0(

)
{
    state_matrix l_ts_x;
    state_matrix l_ts_y;

    const size_t INPUT_WIDTH = 10;
    const size_t FEATURE_VECTOR_WIDTH = 10;
    const size_t IMAGE_WIDTH = 100;
    const std::vector<size_t> FEATURE_MODEL_DIMS = { 20, FEATURE_VECTOR_WIDTH };
    const std::vector<size_t> CHASING_MODEL_DIMS=  { 20, 2 };

    ////////////////////////////////////////////////////////
    // First, create the f model. (Maps from Q1 to F) //////
    ////////////////////////////////////////////////////////

    element_vector::start();
    parameter_vector::start();
    
    auto l_f_x = input(INPUT_WIDTH);
    
    auto l_f_y = pointers(l_f_x);

    for (size_t l_layer_size : FEATURE_MODEL_DIMS)
    {
        l_f_y = weight_junction(l_f_y, l_layer_size);
        l_f_y = bias(l_f_y);
        l_f_y = leaky_relu(l_f_y, 0.3);
    }

    element_vector l_f = element_vector::stop();
    parameter_vector l_f_params = parameter_vector::stop(-1, 1);
    

    ////////////////////////////////////////////////////////
    // Next, create the g model. (Maps from Q2 to F) //////
    ////////////////////////////////////////////////////////
    
    element_vector::start();
    parameter_vector::start();

    auto l_g_x = input(IMAGE_WIDTH);

    auto l_g_y = pointers(l_g_x);

    for (size_t l_layer_size : FEATURE_MODEL_DIMS)
    {
        l_g_y = weight_junction(l_g_y, l_layer_size);
        l_g_y = bias(l_g_y);
        l_g_y = leaky_relu(l_g_y, 0.3);
    }

    element_vector l_g = element_vector::stop();
    parameter_vector l_g_params = parameter_vector::stop(-1, 1);


    ////////////////////////////////////////////////////////
    // Next, define the characteristic loss. ///////////////
    ////////////////////////////////////////////////////////

    element_vector::start();

    auto l_characteristic_loss_y = mean_squared_error(l_f_y, l_g_y)->depend();

    element_vector l_characteristic_loss = element_vector::stop();


    ////////////////////////////////////////////////////////
    // Next, define the chasing models. ////////////////////
    ////////////////////////////////////////////////////////

    element_vector::start();
    parameter_vector::start();

    sgp_ptr_matrix l_h_y(FEATURE_VECTOR_WIDTH);

    // DEFINE THE INPUT MATRIX TO THE CHASING MODELS
    for (int i = 0; i < FEATURE_VECTOR_WIDTH; i++)
    {
        for (int j = 0; j < FEATURE_VECTOR_WIDTH; j++)
        {
            if (i == j)
                continue;
            
            l_h_y[i].push_back(l_f_y[j]);
            l_h_y[i].push_back(l_g_y[j]);

        }
    }

    for (auto& l_h_y_row : l_h_y)
    {
        for (size_t l_layer_size : CHASING_MODEL_DIMS)
        {
            l_h_y_row = weight_junction(l_h_y_row, l_layer_size);
            l_h_y_row = bias(l_h_y_row);
            l_h_y_row = leaky_relu(l_h_y_row, 0.3);
        }
    }

    // F0^ G0^
    // F1^ G1^
    // F2^ G2^
    // F3^ G3^

    auto l_f_prediction = flatten(range(l_h_y, 0, 0, l_h_y.size(), 1));
    auto l_g_prediction = flatten(range(l_h_y, 0, 1, l_h_y.size(), 1));

    auto l_independence_loss = average({
        mean_squared_error(l_f_prediction, l_f_y),
        mean_squared_error(l_g_prediction, l_g_y)
    })->depend();

    element_vector l_h = element_vector::stop();
    parameter_vector l_h_params = parameter_vector::stop(-1, 1);


    ////////////////////////////////////////////////////////
    // Next, we will define the optimizers. ////////////////
    ////////////////////////////////////////////////////////
    gradient_descent_with_momentum l_feature_optimizer(concat(l_f_params, l_g_params), true, 0.02, 0.9);
    gradient_descent_with_momentum l_chasing_optimizer(l_h_params, true, 0.02, 0.9);
    
    
    ////////////////////////////////////////////////////////
    // Next, we will define the training loop. /////////////
    ////////////////////////////////////////////////////////

    for (int l_epoch = 0; true; l_epoch++)
    {
        
    }
    




}

int main(

)
{
	srand(time(0));

	tnn_test();

	return 0;
}
