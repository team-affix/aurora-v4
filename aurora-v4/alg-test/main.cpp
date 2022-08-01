#include "aurora-v4/aurora.h"
#include <iostream>
#include "affix-base/stopwatch.h"

using namespace aurora;

void tnn_test(

)
{
	model::begin();

	std::vector<state_gradient_pair> l_x(2);

	tnn l_tnn(
		pointers(l_x),
		{
			tnn::layer_info(5, neuron_tanh()),
			tnn::layer_info(1, neuron_sigmoid())
		}
	);

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

		double l_cost = mean_squared_error(l_tnn.m_y, a_y);
			
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
			std::cout << l_tnn.m_y[0]->m_state << std::endl;

		l_cost += l_cycle(l_ts_x[1], l_ts_y[1]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_tnn.m_y[0]->m_state << std::endl;

		l_cost += l_cycle(l_ts_x[2], l_ts_y[2]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_tnn.m_y[0]->m_state << std::endl;

		l_cost += l_cycle(l_ts_x[3], l_ts_y[3]);

		if (epoch % CHECKPOINT == 0)
			std::cout << l_tnn.m_y[0]->m_state << std::endl;

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

	tnn l_tnn(
		pointers(l_x),
		{
			tnn::layer_info(20, neuron_leaky_relu()),
			tnn::layer_info(1, neuron_leaky_relu())
		});

	model l_model = model::end();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(25);

	for (int i = 0; i < l_model.parameters().size(); i++)
	{
		l_model.parameters()[i]->m_state = l_urd(l_dre);
	}

	auto l_cycle = [&](const std::vector<state_gradient_pair>& a_x, const std::vector<state_gradient_pair>& a_y)
	{
		set_state(l_x, a_x);

		l_model.fwd();

		double l_cost = mean_squared_error(l_tnn.m_y, a_y);

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
				std::cout << "INPUT: " << l_ts_x << ", PREDICTION: " << l_tnn.m_y[0]->m_state << ", DESIRED: " << l_ts_y << std::endl;

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

	affix_base::data::ptr<multiply> l_multiply(new multiply(&l_x_0, &l_x_1));

	affix_base::data::ptr<branch> l_branch(new branch(model::end(), true));
	
	model l_model = model::end();

	l_model.fwd();

}

void lstm_test(

)
{
	model::begin();

	std::vector<affix_base::data::ptr<gradient_descent>> l_optimizers;

	const size_t l_lstm_y_units = 1;
	const size_t l_tnn_h0_units = 3;
	const size_t l_tnn_y_units = 1;

	auto l_x = matrix(4, 2);

	lstm l_lstm_0(pointers(l_x), l_lstm_y_units);

	std::vector<std::vector<state_gradient_pair*>> l_y;

	for (int i = 0; i < l_lstm_0.m_y.size(); i++)
	{
		tnn l_tnn(
			l_lstm_0.m_y[i],
			{
				tnn::layer_info(l_tnn_h0_units, neuron_leaky_relu()),
				tnn::layer_info(l_tnn_y_units, neuron_sigmoid())
			});
		l_y.push_back(l_tnn.m_y);
	}

	model l_model = model::end();

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(28);

	for (auto& l_parameter : l_model.parameters())
	{
		l_parameter->m_state = l_urd(l_dre);
		l_optimizers.push_back(new gradient_descent(l_parameter, 0.2));
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

		for (int i = 0; i < l_optimizers.size(); i++)
			l_optimizers[i]->update();

		if (epoch % CHECKPOINT == 0)
			std::cout << "COST: " << l_cost << std::endl << std::endl;

	}

}

void lstm_stacked_test(

)
{
	model::begin();

	auto l_x = matrix(4, 2);

	lstm l_lstm_0(pointers(l_x), 20);
	lstm l_lstm_1(l_lstm_0.m_y, 20);
	lstm l_lstm_2(l_lstm_1.m_y, 1);

	std::vector<std::vector<state_gradient_pair*>> l_y;

	for (int i = 0; i < l_lstm_2.m_y.size(); i++)
	{
		tnn l_tnn(
			l_lstm_2.m_y[i],
			{
				tnn::layer_info(5, neuron_leaky_relu()),
				tnn::layer_info(1, neuron_sigmoid())
			});
		l_y.push_back(l_tnn.m_y);
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

int main(

)
{
	srand(time(0));

	parabola_test();

	return 0;
}
