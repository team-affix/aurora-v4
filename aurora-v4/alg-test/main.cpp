#include "aurora-v4/aurora.h"
#include <iostream>
#include "affix-base/stopwatch.h"
#include "affix-base/vector_extensions.h"
#include "cryptopp/osrng.h"
#include "affix-base/persistent_thread.h"

using namespace aurora;
using namespace aurora::latent;

void tnn_test(

)
{
	element_vector::start();
	parameter_vector::start();

	auto l_x = input(4, 2);
	auto l_desired_y = input(4, 1);

	std::vector<std::vector<state_gradient_pair*>> l_y;

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
	
	std::vector<state_gradient_pair*> l_cross_entropy_losses;

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

	std::vector<std::vector<double>> l_ts_x =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	std::vector<std::vector<double>> l_ts_y =
	{
		{0},
		{1},
		{1},
		{0}
	};

	affix_base::timing::stopwatch l_stopwatch;

	l_stopwatch.start();

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

	std::cout << std::endl << "PERIOD OF TRAINING (ms): " << l_stopwatch.duration_milliseconds() << std::endl;

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
	
	std::vector<std::vector<state_gradient_pair*>> l_y(l_x.size());

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

		std::vector<std::vector<double>> l_ts_x;
		std::vector<std::vector<double>> l_ts_y;

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

	std::vector<std::vector<std::vector<state_gradient_pair*>>> l_y;

	size_t l_model_begin_next_parameter_index = parameter_vector::next_index();

	for (int i = 0; i < l_x.size(); i++)
	{
		parameter_vector::next_index(l_model_begin_next_parameter_index);
		auto l_lstm_y = lstm(pointers(l_x[i]), l_lstm_y_units);
		size_t l_tnn_begin_next_parameter_index = parameter_vector::next_index();
		std::vector<std::vector<state_gradient_pair*>> l_tnn_ys;
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

	std::vector<std::vector<std::vector<double>>> l_training_set_xs =
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

	std::vector<std::vector<std::vector<double>>> l_training_set_ys =
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

	std::vector<std::vector<state_gradient_pair*>> l_y;

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

	std::vector<std::vector<std::vector<double>>> l_training_set_xs =
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

	std::vector<std::vector<std::vector<double>>> l_training_set_ys =
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

	std::vector<std::vector<state_gradient_pair>> l_x_0
	{
		{1, 2},
		{3, 4},
		{5, 6},
		{7, 8}
	};

	std::vector<state_gradient_pair> l_x_1
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

	std::vector<state_gradient_pair> l_x_0{ 0, 1, 0, 0 };
	std::vector<state_gradient_pair> l_x_1{ 0, -1, 0, 0 };

	auto l_y = cosine_similarity(pointers(l_x_0), pointers(l_x_1));

	auto l_model = element_vector::stop();

	l_model.fwd();

}

void similarity_interpolate_test(

)
{
	std::vector<std::vector<state_gradient_pair>> l_tsx =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{1, 0.75}
	};

	std::vector<std::vector<state_gradient_pair>> l_tsy =
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

	std::vector<state_gradient_pair> l_x(1000);

	affix_base::timing::stopwatch l_stopwatch;
	l_stopwatch.start();

	{
		auto l_y = pointers(l_x);
		l_y = weight_junction(l_y, 1000);
		l_y = weight_junction(l_y, 1000);
		l_y = weight_junction(l_y, 1000);

		element_vector l_model = element_vector::stop();
		auto l_parameters = parameter_vector::stop();

		std::cout << "MODEL CREATED: " << l_model.size() << " elements; " << l_stopwatch.duration_milliseconds() << " ms" << std::endl;
		l_stopwatch.start();

		std::uniform_real_distribution<double> l_urd(-1, 1);
		std::default_random_engine l_dre(25);

		for (auto& l_parameter : l_parameters)
			l_parameter->m_state = l_urd(l_dre);
		std::cout << "PARAMETERS INITIALIZED: " << l_stopwatch.duration_milliseconds() << " ms" << std::endl;
		l_stopwatch.start();

		l_model.fwd();
		std::cout << "FORWARD COMPLETED: " << l_stopwatch.duration_milliseconds() << " ms" << std::endl;
		l_stopwatch.start();

		l_model.bwd();
		std::cout << "BACKWARD COMPLETED: " << l_stopwatch.duration_milliseconds() << " ms" << std::endl;
		l_stopwatch.start();
	}

	std::cout << "DECONSTRUCTED: " << l_stopwatch.duration_milliseconds() << " ms" << std::endl;


}

std::vector<std::vector<std::vector<state_gradient_pair*>>> in_sequence_stock_predict(
	std::vector<std::vector<state_gradient_pair*>> a_x,
	const std::vector<size_t>& a_lstm_y_sizes,
	const std::vector<size_t>& a_layer_y_sizes,
	const size_t& a_time_slots_to_predict_per_timestep,
	const size_t& a_time_slot_prediction_size
)
{
	std::vector<std::vector<state_gradient_pair*>> l_y_raw = a_x;

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

	std::vector<std::vector<std::vector<state_gradient_pair*>>> l_future_hour_predictions;

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

	std::vector<std::vector<state_gradient_pair>> l_x = input(100, 4);

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
	std::vector<state_gradient_pair> l_x = { 0, 0 };

	std::vector<state_gradient_pair*> l_y = pointers(l_x);

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

	std::vector<std::vector<double>> l_tsx =
	{
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
	};

	std::vector<std::vector<double>> l_tsy =
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
	std::vector<state_gradient_pair> l_x(3);

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
		std::vector<double> m_x;
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
	std::vector<state_gradient_pair> l_task_x(10);
	std::vector<state_gradient_pair> l_task_prediction(1);
	std::vector<state_gradient_pair> l_loss_model_desired_y(1);

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
	std::vector<std::vector<double>> l_tsx =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	std::vector<std::vector<double>> l_tsy =
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

void oneshot_tnn_test(

)
{
	std::vector<affix_base::threading::persistent_thread> l_threads(std::thread::hardware_concurrency() / 2);
	std::vector<std::function<void()>> l_functions(l_threads.size());
	std::vector<std::vector<double>> l_x(l_threads.size());
	std::vector<std::vector<double>> l_y(l_threads.size());
	std::vector<oneshot::parameter_vector> l_parameter_vectors;

	oneshot::parameter_vector l_parameters(-1, 1);

	std::function<std::vector<double>(oneshot::parameter_vector&, const std::vector<double>&)> l_carry_forward = 
		[](oneshot::parameter_vector& a_parameter_vector, const std::vector<double>& a_x)
	{
		auto l_y = a_x;

		l_y = oneshot::multiply(a_parameter_vector.next(5, l_y.size()), l_y);
		l_y = oneshot::add(l_y, a_parameter_vector.next(5));
		l_y = oneshot::leaky_relu(l_y, 0.3);
		l_y = oneshot::multiply(a_parameter_vector.next(1, l_y.size()), l_y);
		l_y = oneshot::add(l_y, a_parameter_vector.next(1));
		l_y = oneshot::sigmoid(l_y);

		return l_y;

	};

	auto l_dry_fire = l_carry_forward(l_parameters, { 0, 0 });

}

int main(

)
{
	srand(time(0));

	oneshot_tnn_test();

	return 0;
}
