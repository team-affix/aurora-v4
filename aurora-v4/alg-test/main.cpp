#include "aurora-v4/aurora.h"
#include <iostream>
#include "affix-base/stopwatch.h"

using namespace aurora;

void tnn_test(

)
{
	model::begin();

	std::vector<state_gradient_pair> l_x(2);

	auto l_y = pointers(l_x);
	l_y = weight_junction(l_y, 5);
	l_y = bias(l_y);
	l_y = leaky_relu(l_y, 0.3);
	l_y = weight_junction(l_y, 1);
	l_y = bias(l_y);
	l_y = sigmoid(l_y);

	model l_model = model::end();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(25);

	for (int i = 0; i < l_model.parameters().size(); i++)
	{
		l_model.parameters()[i]->m_state = l_urd(l_dre);
	}

	auto l_cycle = [&](const std::vector<state_gradient_pair>& a_x, const std::vector<state_gradient_pair>& a_y)
	{
		for (int i = 0; i < l_x.size(); i++)
			l_x[i].m_state = a_x[i].m_state;

		l_model.fwd();

		double l_cost = mean_squared_error(l_y, a_y);
			
		l_model.bwd();

		return l_cost;

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

		for (int i = 0; i < l_model.parameters().size(); i++)
		{
			l_model.parameters()[i]->m_state -= 0.002 * l_model.parameters()[i]->m_gradient;
			l_model.parameters()[i]->m_gradient = 0;
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

	model::begin();

	auto l_y = pointers(l_x);
	l_y = weight_junction(l_y, 20);
	l_y = bias(l_y);
	l_y = leaky_relu(l_y, 0.3);
	l_y = weight_junction(l_y, 1);
	l_y = bias(l_y);
	l_y = leaky_relu(l_y, 0.3);

	model l_model = model::end(-1, 1);

	std::default_random_engine l_dre(25);

	auto l_cycle = [&](const std::vector<state_gradient_pair>& a_x, const std::vector<state_gradient_pair>& a_y)
	{
		set_state(l_x, a_x);

		l_model.fwd();

		double l_cost = mean_squared_error(l_y, a_y);

		l_model.bwd();

		return l_cost;

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
			l_cost += l_cycle({ l_ts_x }, { l_ts_y });

			if (epoch % CHECKPOINT_INTERVAL == 0)
				std::cout << "INPUT: " << l_ts_x << ", PREDICTION: " << l_y[0]->m_state << ", DESIRED: " << l_ts_y << std::endl;

		}

		l_cost_momentum = 0.99 * l_cost_momentum + 0.01 * l_cost;

		for (int i = 0; i < l_model.parameters().size(); i++)
		{
			l_model.parameters()[i]->m_state -= 0.002 * tanh(l_model.parameters()[i]->m_gradient);
			l_model.parameters()[i]->m_gradient = 0;
		}

		if (epoch % CHECKPOINT_INTERVAL == 0)
			std::cout << "    LOSS FOR ABOVE EPOCH: " << l_cost_momentum << std::endl;

	}

}

void branch_test(

)
{
	model::begin();

	state_gradient_pair l_x_0 = { 1.5 };
	state_gradient_pair l_x_1 = { 2 };


	// Start a new model for the branch
	model::begin();

	state_gradient_pair* l_multiplied = multiply(&l_x_0, &l_x_1);

	bool* l_branch_enabled = branch(model::end(), true);
	
	model l_model = model::end();

	l_model.fwd();

}

void lstm_test(

)
{
	model::begin();

	const size_t l_lstm_y_units = 3;
	const size_t l_tnn_h0_units = 3;
	const size_t l_tnn_y_units = 1;

	auto l_x = matrix(4, 2);

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

	model l_model = model::end(-1, 1, gradient_descent(0.2));

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
			set_state(l_x, l_training_set_xs[i]);
			
			// Carry forward
			l_model.fwd();

			// Signal output
			l_cost += mean_squared_error(l_y, l_training_set_ys[i]);

			// Carry backward
			l_model.bwd();

			if (epoch % CHECKPOINT == 0)
			{
				for (int i = 0; i < l_y.size(); i++)
					std::cout << "PREDICTION: " << l_y[i][0]->m_state << std::endl;
				std::cout << std::endl;
			}

		}

		l_model.update();

		if (epoch % CHECKPOINT == 0)
			std::cout << "COST: " << l_cost << std::endl << std::endl;

	}

}

void lstm_stacked_test(

)
{
	model::begin();

	auto l_x = matrix(4, 2);

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

	model l_model = model::end();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(28);

	for (auto& l_parameter : l_model.parameters())
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
			set_state(l_x, l_training_set_xs[i]);
			l_model.fwd();
			l_cost += mean_squared_error(l_y, l_training_set_ys[i]);
			l_model.bwd();
		}

		for (auto& l_parameter : l_model.parameters())
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
	model::begin();

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

	auto l_y = matrix_vector_multiply(pointers(l_x_0), pointers(l_x_1));

	model l_model = model::end();

	l_model.fwd();

}

void cosine_similarity_test(

)
{
	model::begin();

	std::vector<state_gradient_pair> l_x_0{ 0, 1, 0, 0 };
	std::vector<state_gradient_pair> l_x_1{ 0, -1, 0, 0 };

	auto l_y = cosine_similarity(pointers(l_x_0), pointers(l_x_1));

	model l_model = model::end();

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

	std::vector<state_gradient_pair> l_query = { 1, 0 };
	
	model::begin();

	auto l_y = similarity_interpolate(pointers(l_query), pointers(l_tsx), pointers(l_tsy));

	model l_model = model::end();

	l_model.fwd();

}

void large_memory_usage_test(

)
{
	model::begin();

	std::vector<state_gradient_pair> l_x(1000);

	affix_base::timing::stopwatch l_stopwatch;
	l_stopwatch.start();

	{
		auto l_y = pointers(l_x);
		l_y = weight_junction(l_y, 1000);
		l_y = weight_junction(l_y, 1000);
		l_y = weight_junction(l_y, 1000);

		model l_model = model::end();
		std::cout << "MODEL CREATED: " << l_model.elements().size() << " elements; " << l_stopwatch.duration_milliseconds() << " ms" << std::endl;
		l_stopwatch.start();

		std::uniform_real_distribution<double> l_urd(-1, 1);
		std::default_random_engine l_dre(25);

		for (auto& l_parameter : l_model.parameters())
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
	model::begin();

	std::vector<std::vector<state_gradient_pair>> l_x = matrix(100, 4);

	auto l_y = in_sequence_stock_predict(
		pointers(l_x),
		{ 20, 20 },
		{ 20, 40 },
		10,
		2
	);

	model l_model = model::end(-1, 1, gradient_descent(0.02));

	l_model.fwd();



	l_model.bwd();



}

void pablo_tnn_example(

)
{
	model::begin();

	// Write model building code here
	std::vector<state_gradient_pair> l_x = { 0, 0 };

	std::vector<state_gradient_pair*> l_y = pointers(l_x);

	l_y = weight_junction(l_y, 5);
	l_y = bias(l_y);
	l_y = tanh(l_y);
	
	l_y = weight_junction(l_y, 1);
	l_y = bias(l_y);
	l_y = sigmoid(l_y);

							 // "model" is just a structure which holds a vector of elements and a vector of parameters.
							 // This function call finalizes our model and spits it out. (This also initializes parameters)
	model l_model = model::end(-1, 1, gradient_descent_with_momentum(0.02, 0.9));

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
			set_state(l_x, l_tsx[i]);
			l_model.fwd();
			mean_squared_error(l_y, l_tsy[i]);
			l_model.bwd();
			if (epoch % CHECKPOINT == 0)
			{
				std::cout << l_y[0]->m_state << std::endl;
			}
		}

		l_model.update();

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

	model::begin();

	auto l_y = parameterized_interpolate(
		pointers(l_x)
	);

	model::end(-1, 1, gradient_descent(0.02));

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

}

int main(

)
{
	srand(time(0));

	parabola_test();

	return 0;
}
