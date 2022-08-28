#include "aurora-v4/aurora.h"
#include <iostream>
#include "affix-base/stopwatch.h"
#include "affix-base/vector_extensions.h"
#include "cryptopp/osrng.h"

using namespace aurora;

void tnn_test(

)
{
	element_vector::start();
	parameter_vector::start();

	std::vector<state_gradient_pair> l_x(2);

	auto l_y = pointers(l_x);
	l_y = weight_junction(l_y, 5);
	l_y = bias(l_y);
	l_y = leaky_relu(l_y, 0.3);
	l_y = weight_junction(l_y, 1);
	l_y = bias(l_y);
	l_y = sigmoid(l_y);
	
	auto l_desired_y = input(l_y.size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y));
	
	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(25);

	for (int i = 0; i < l_parameters.size(); i++)
	{
		l_parameters[i]->m_state = l_urd(l_dre);
	}

	auto l_cycle = [&](std::vector<state_gradient_pair>& a_x, std::vector<state_gradient_pair>& a_y)
	{
		set_state(pointers(l_x), pointers(a_x));
		set_state(pointers(l_desired_y), pointers(a_y));

		l_model.fwd();

		l_error->m_gradient = 1;
			
		l_model.bwd();

		return l_error->m_state;

	};

	const int CHECKPOINT = 100000;

	std::vector<std::vector<state_gradient_pair>> l_ts_x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	std::vector<std::vector<state_gradient_pair>> l_ts_y = {
		{0},
		{1},
		{1},
		{0}
	};

	affix_base::timing::stopwatch l_stopwatch;
	l_stopwatch.start();

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		double l_cost = 0;

		l_cost += l_cycle(l_ts_x[0], l_ts_y[0]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[0]->m_state << std::endl;

		l_cost += l_cycle(l_ts_x[1], l_ts_y[1]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[0]->m_state << std::endl;

		l_cost += l_cycle(l_ts_x[2], l_ts_y[2]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[0]->m_state << std::endl;

		l_cost += l_cycle(l_ts_x[3], l_ts_y[3]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_y[0]->m_state << std::endl;

		if (epoch % CHECKPOINT == 0)
			std::cout << std::endl;

		for (int i = 0; i < l_parameters.size(); i++)
		{
			l_parameters[i]->m_state -= 0.002 * l_parameters[i]->m_gradient;
			l_parameters[i]->m_gradient = 0;
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
	std::vector<state_gradient_pair> l_x(1);

	element_vector::start();
	parameter_vector::start();
	
	auto l_y = pointers(l_x);
	l_y = weight_junction(l_y, 20);
	l_y = bias(l_y);
	l_y = leaky_relu(l_y, 0.3);
	l_y = weight_junction(l_y, 1);
	l_y = bias(l_y);
	l_y = leaky_relu(l_y, 0.3);

	auto l_desired_y = input(l_y.size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y));

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	std::default_random_engine l_dre(25);

	auto l_cycle = [&](std::vector<state_gradient_pair>& a_x, std::vector<state_gradient_pair>& a_y)
	{
		set_state(pointers(l_x), pointers(a_x));
		set_state(pointers(l_desired_y), pointers(a_y));

		l_model.fwd();

		l_error->m_gradient = 1;

		l_model.bwd();

		return l_error->m_state;

	};

	std::uniform_real_distribution<double> l_ts_urd(-10, 10);

	double l_cost_momentum = 0;

	const size_t CHECKPOINT_INTERVAL = 10000;

	for (int epoch = 0; true; epoch++)
	{
		double l_cost = 0;

		for (int i = 0; i < 10; i++)
		{
			double l_ts_x = l_ts_urd(l_dre);
			double l_ts_y = l_ts_x * l_ts_x;

			std::vector<state_gradient_pair> l_ts_x_vector = { l_ts_x };
			std::vector<state_gradient_pair> l_ts_y_vector = { l_ts_y };

			l_cost += l_cycle(l_ts_x_vector, l_ts_y_vector);

			if (epoch % CHECKPOINT_INTERVAL == 0)
				std::cout << "INPUT: " << l_ts_x << ", PREDICTION: " << l_y[0]->m_state << ", DESIRED: " << l_ts_y << std::endl;

		}

		l_cost_momentum = 0.99 * l_cost_momentum + 0.01 * l_cost;

		for (int i = 0; i < l_parameters.size(); i++)
		{
			l_parameters[i]->m_state -= 0.002 * tanh(l_parameters[i]->m_gradient);
			l_parameters[i]->m_gradient = 0;
		}

		if (epoch % CHECKPOINT_INTERVAL == 0)
			std::cout << "    LOSS FOR ABOVE EPOCH: " << l_cost_momentum << std::endl;

	}

}

void branch_test(

)
{
	element_vector::start();

	state_gradient_pair l_x_0 = { 1.5 };
	state_gradient_pair l_x_1 = { 2 };


	// Start a new model for the branch
	element_vector::start();

	state_gradient_pair* l_multiplied = multiply(&l_x_0, &l_x_1);

	bool* l_branch_enabled = branch(element_vector::stop(), true);
	
	element_vector l_model = element_vector::stop();

	l_model.fwd();

}

void lstm_test(

)
{
	element_vector::start();
	parameter_vector::start();

	const size_t l_lstm_y_units = 3;
	const size_t l_tnn_h0_units = 3;
	const size_t l_tnn_y_units = 1;

	auto l_x = input(4, 2);

	auto l_lstm_0 = lstm(pointers(l_x), l_lstm_y_units);

	std::vector<std::vector<state_gradient_pair*>> l_y;

	for (int i = 0; i < l_lstm_0.size(); i++)
	{
		auto l_tnn_y = l_lstm_0[i];
		l_tnn_y = weight_junction(l_tnn_y, l_tnn_h0_units);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
		l_tnn_y = weight_junction(l_tnn_y, l_tnn_y_units);
		l_tnn_y = bias(l_tnn_y);
		l_tnn_y = leaky_relu(l_tnn_y, 0.3);
		l_y.push_back(l_tnn_y);
	}

	auto l_desired_y = input(l_y.size(), l_y[0].size());
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y));

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, 0.02);

	std::vector<std::vector<std::vector<state_gradient_pair>>> l_training_set_xs =
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

	std::vector<std::vector<std::vector<state_gradient_pair>>> l_training_set_ys =
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

		for (int i = 0; i < l_training_set_xs.size(); i++)
		{
			set_state(pointers(l_x), pointers(l_training_set_xs[i]));
			set_state(pointers(l_desired_y), pointers(l_training_set_ys[i]));
			
			// Carry forward
			l_model.fwd();

			// Signal output
			l_cost += l_error->m_state;
			l_error->m_gradient = 1;

			// Carry backward
			l_model.bwd();

			if (epoch % CHECKPOINT == 0)
			{
				for (int i = 0; i < l_y.size(); i++)
					std::cout << "PREDICTION: " << l_y[i][0]->m_state << std::endl;
				std::cout << std::endl;
			}

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
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y));

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(28);

	for (auto& l_parameter : l_parameters)
	{
		l_parameter->m_state = l_urd(l_dre);
	}

	std::vector<std::vector<std::vector<state_gradient_pair>>> l_training_set_xs =
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

	std::vector<std::vector<std::vector<state_gradient_pair>>> l_training_set_ys =
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
			set_state(pointers(l_x), pointers(l_training_set_xs[i]));
			set_state(pointers(l_desired_y), pointers(l_training_set_ys[i]));

			l_model.fwd();
			l_cost += l_error->m_state;
			l_error->m_gradient = 1;
			l_model.bwd();
		}

		for (auto& l_parameter : l_parameters)
		{
			l_parameter->m_state -= 0.2 * l_parameter->m_gradient;
			l_parameter->m_gradient = 0;
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
	auto l_error = mean_squared_error(l_y, pointers(l_desired_y));

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, 0.02);

	std::vector<std::vector<state_gradient_pair>> l_tsx =
	{
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 },
	};

	std::vector<std::vector<state_gradient_pair>> l_tsy =
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
			set_state(pointers(l_x), pointers(l_tsx[i]));
			set_state(pointers(l_desired_y), pointers(l_tsy[i]));

			l_model.fwd();
			l_error->m_gradient = 1;
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
	auto l_error = mean_squared_error(l_y, &l_desired_y);

	auto l_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, 0.02);

	struct training_set
	{
		std::vector<state_gradient_pair> m_x;
		state_gradient_pair m_y;
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
			set_state(pointers(l_x), pointers(l_training_set.m_x));
			l_desired_y.m_state = l_training_set.m_y.m_state;

			l_model.fwd();

			l_cost += l_error->m_state;
			l_error->m_gradient = 1;

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

	auto l_loss_model_loss = mean_squared_error(l_loss_model_y, pointers(l_loss_model_desired_y));

	auto l_loss_model = element_vector::stop();
	auto l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, 0.02);

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
			
			l_loss_model_loss->m_gradient = 1;
			l_loss_model_epoch_loss += l_loss_model_loss->m_state;

			l_loss_model.bwd();

		}

		l_optimizer.normalize_gradients();
		l_optimizer.update();

		if (epoch % 10000 == 0)
		{
			std::cout << "LR: " << l_optimizer.m_learn_rate << ", LOSS: " << l_loss_model_epoch_loss << std::endl;
			l_optimizer.m_learn_rate *= 0.99;
		}

	}

	std::cout << std::endl << std::endl << "GENERATING TASK PREDICTION: " << std::endl;

	gradient_descent l_task_prediction_optimizer(pointers(l_task_prediction), 0.2);
	 
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
		l_loss_model_loss->m_gradient = 1;
		l_loss_model.bwd();
		l_task_prediction_optimizer.normalize_gradients();
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

	gradient_descent l_task_x_optimizer(pointers(l_task_x), 0.2);

	l_task_prediction[0].m_state = 10000;
	l_loss_model_desired_y[0].m_state = 0;

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		l_loss_model.fwd();
		l_loss_model_loss->m_gradient = 1;
		l_loss_model.bwd();
		l_task_x_optimizer.normalize_gradients();
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
	std::vector<std::vector<state_gradient_pair>> l_tsx =
	{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	std::vector<std::vector<state_gradient_pair>> l_tsy =
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
	auto l_loss = mean_squared_error(l_y, pointers(l_desired));

	element_vector l_elements = element_vector::stop();
	parameter_vector l_parameters = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_parameters, 0.02);

	const int CHECKPOINT = 100000;

	for (int epoch = 0; true; epoch++)
	{
		for (int i = 0; i < l_tsx.size(); i++)
		{
			set_state(pointers(l_x), pointers(l_tsx[i]));
			set_state(pointers(l_desired), pointers(l_tsy[i]));
			l_elements.fwd();
			l_loss->m_gradient = 1;
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

void teacher_student_test_0(

)
{
	const size_t STUDENT_TASKS_COUNT = 100;
	const size_t STUDENT_TRAINING_SETS_PER_TASK_COUNT = 10;
	const size_t STUDENT_TESTING_SETS_PER_TASK_COUNT = 10;
	const size_t STUDENT_INPUT_SIZE = 128;
	const std::vector<size_t> STUDENT_DIMENSIONS = { 32, 32 };
	const double STUDENT_MINIMUM_IO_VALUE = -100;
	const double STUDENT_MAXIMUM_IO_VALUE = 100;
	const size_t MINI_BATCH_SIZE = 32;

	// CREATE STUDENT
	element_vector::start();
	parameter_vector::start();

	auto l_student_x = input(STUDENT_INPUT_SIZE);
	auto l_student_y = pointers(l_student_x);

	for (int i = 0; i < STUDENT_DIMENSIONS.size(); i++)
	{
		l_student_y = weight_junction(l_student_y, STUDENT_DIMENSIONS[i]);
		l_student_y = bias(l_student_y);
		l_student_y = leaky_relu(l_student_y, 0.3);
	}

	auto l_student_desired_y = input(l_student_y.size());
	auto l_student_loss = mean_squared_error(l_student_y, pointers(l_student_desired_y));

	element_vector l_student_element_vector = element_vector::stop();
	parameter_vector l_student_parameter_vector = parameter_vector::stop();

	// GENERATE STUDENT TRAINING SETS
	
	struct student_training_set
	{
		std::vector<state_gradient_pair> m_x;
		std::vector<state_gradient_pair> m_y;
	};

	struct teacher_training_set
	{
		std::vector<std::vector<state_gradient_pair>> m_x;
		std::vector<student_training_set> m_student_testing_sets;
	};
	
	std::vector<teacher_training_set> l_teacher_training_sets;

	for (int i = 0; i < STUDENT_TASKS_COUNT; i++)
	{
		// RANDOMIZE STUDENT PARAM VECTOR
		randomize_state(l_student_parameter_vector, -10, 10);

		teacher_training_set l_teacher_training_set;

		for (int j = 0; j < STUDENT_TRAINING_SETS_PER_TASK_COUNT; j++)
		{
			randomize_state(pointers(l_student_x), STUDENT_MINIMUM_IO_VALUE, STUDENT_MAXIMUM_IO_VALUE);
			l_student_element_vector.fwd();
			l_teacher_training_set.m_x.push_back(get_state(concat(pointers(l_student_x), l_student_y)));
		}

		for (int j = 0; j < STUDENT_TESTING_SETS_PER_TASK_COUNT; j++)
		{
			randomize_state(pointers(l_student_x), STUDENT_MINIMUM_IO_VALUE, STUDENT_MAXIMUM_IO_VALUE);
			l_student_element_vector.fwd();
			l_teacher_training_set.m_student_testing_sets.push_back({l_student_x, get_state(l_student_y)});
		}

		l_teacher_training_sets.push_back(l_teacher_training_set);

	}

	const std::vector<size_t> TEACHER_DIMENSIONS = { 128, l_student_parameter_vector.size()};

	// CREATE TEACHER
	element_vector::start();
	parameter_vector::start();

	auto l_teacher_x = input(STUDENT_TRAINING_SETS_PER_TASK_COUNT, STUDENT_INPUT_SIZE + STUDENT_DIMENSIONS.back());
	auto l_teacher_y = pointers(l_teacher_x);

	for (int i = 0; i < TEACHER_DIMENSIONS.size(); i++)
		l_teacher_y = lstm(l_teacher_y, TEACHER_DIMENSIONS[i]);

	element_vector l_teacher_element_vector = element_vector::stop();
	parameter_vector l_teacher_parameter_vector = parameter_vector::stop(-1, 1);

	gradient_descent l_optimizer(l_teacher_parameter_vector, 0.02);

	CryptoPP::AutoSeededRandomPool l_random;

	for (int epoch = 0; true; epoch++)
	{
		double l_cost = 0;

		for (int i = 0; i < MINI_BATCH_SIZE; i++)
		{
			size_t l_training_set_index = l_random.GenerateWord32(0, l_teacher_training_sets.size());
			teacher_training_set& l_teacher_training_set = l_teacher_training_sets[l_training_set_index];
			
			set_state(pointers(l_teacher_x), pointers(l_teacher_training_set.m_x));
			
			l_teacher_element_vector.fwd();
			
			for (int l_timestep = 0; l_timestep < l_teacher_y.size(); l_timestep++)
			{
				// SET PARAMETERS OF STUDENT
				set_state(l_student_parameter_vector, l_teacher_y[l_timestep]);

				for (auto& l_student_training_set : l_teacher_training_set.m_student_testing_sets)
				{
					// CYCLE THE STUDENT
					set_state(pointers(l_student_x), pointers(l_student_training_set.m_x));
					set_state(pointers(l_student_desired_y), pointers(l_student_training_set.m_y));
					l_student_element_vector.fwd();
					l_cost += l_student_loss->m_state;
					l_student_loss->m_gradient = 1;
					l_student_element_vector.bwd();
				}

				add_gradient(l_teacher_y[l_timestep], l_student_parameter_vector);
				clear_gradient(l_student_parameter_vector);

			}

			l_teacher_element_vector.bwd();

		}


		l_optimizer.normalize_gradients();
		l_optimizer.update();

		if (epoch % 10 == 0)
			std::cout << l_cost << std::endl;

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

int main(

)
{
	srand(time(0));

	tnn_test();

	return 0;
}
