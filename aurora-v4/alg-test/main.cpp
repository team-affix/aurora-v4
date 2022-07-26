#include "aurora-v4/aurora.h"
#include <iostream>
#include "affix-base/stopwatch.h"

using namespace aurora;

void additive_aggregate_test(

)
{
	std::cout << "ADDITIVE AGGREGATE TEST" << std::endl;
	std::vector<affix_base::data::ptr<element>> l_elements;
	std::vector<parameter*> l_parameters;

	std::vector<state_gradient_pair> l_x = { {1}, {2}, {3}, {4} };

	additive_aggregate l_aggregate(l_elements, { &l_x[0], &l_x[1], &l_x[2], &l_x[3] });

	for (int epoch = 0; epoch < 10000; epoch++)
	{
		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		l_aggregate.m_y->m_gradient = l_aggregate.m_y->m_state - 20.0;

		if (epoch % 100 == 0)
			std::cout << l_aggregate.m_y->m_gradient << std::endl;

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		for (int i = 0; i < l_x.size(); i++)
		{
			l_x[i].m_state -= 0.002 * l_x[i].m_gradient;
			l_x[i].m_gradient = 0;
		}
	}

}

void multiplicative_aggregate_test(

)
{
	std::cout << "MULTIPLICATIVE AGGREGATE TEST" << std::endl;
	std::vector<affix_base::data::ptr<element>> l_elements;
	std::vector<parameter*> l_parameters;

	std::vector<state_gradient_pair> l_x = { {1}, {2}, {3}, {4} };

	multiplicative_aggregate l_aggregate(l_elements, { &l_x[0], &l_x[1], &l_x[2], &l_x[3] });

	for (int epoch = 0; epoch < 1000; epoch++)
	{
		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		l_aggregate.m_y->m_gradient = l_aggregate.m_y->m_state - 20.0;

		if (epoch % 100 == 0)
			std::cout << l_aggregate.m_y->m_gradient << std::endl;

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		for (int i = 0; i < l_x.size(); i++)
		{
			l_x[i].m_state -= 0.002 * l_x[i].m_gradient;
			l_x[i].m_gradient = 0;
		}

	}

}

void tanh_activate_test(

)
{
	std::vector<affix_base::data::ptr<element>> l_elements;

	state_gradient_pair l_x = { 3 };

	affix_base::data::ptr<tanh_activate> l_tanh_activate(new tanh_activate(l_elements, &l_x));

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		l_tanh_activate->m_y.m_gradient = l_tanh_activate->m_y.m_state - (-0.5);

		if (epoch % 100 == 0)
			std::cout << l_tanh_activate->m_y.m_gradient << std::endl;

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		l_x.m_state -= 0.02 * l_x.m_gradient;
		l_x.m_gradient = 0;
	}

}

void leaky_relu_activate_test(

)
{
	std::vector<affix_base::data::ptr<element>> l_elements;

	state_gradient_pair l_x = { 3 };

	affix_base::data::ptr<leaky_relu_activate> l_leaky_relu_activate(new leaky_relu_activate(l_elements, &l_x, 0.3));

	for (int epoch = 0; epoch < 100000; epoch++)
	{
		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		l_leaky_relu_activate->m_y.m_gradient = l_leaky_relu_activate->m_y.m_state + 20;

		if (epoch % 100 == 0)
			std::cout << l_leaky_relu_activate->m_y.m_gradient << std::endl;

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		l_x.m_state -= 0.02 * l_x.m_gradient;
		l_x.m_gradient = 0;
	}

}

void normalize_test(

)
{
	std::vector<affix_base::data::ptr<element>> l_elements;

	std::vector<state_gradient_pair> l_x = { {1}, {2}, {3} };

	normalize l_normalize(l_elements, { &l_x[0], &l_x[1], &l_x[2] });

	for (int epoch = 0; epoch < 10000; epoch++)
	{
		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		double l_cost = 0;

		for (int i = 0; i < l_normalize.m_y.size(); i++)
		{
			l_normalize.m_y[i]->m_gradient = l_normalize.m_y[i]->m_state - 0.333333333333;
			l_cost += abs(l_normalize.m_y[i]->m_gradient);
		}

		if (epoch % 100 == 0)
			std::cout << l_cost << std::endl;

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		for (int i = 0; i < l_x.size(); i++)
			l_x[i].m_state -= 0.02 * l_x[i].m_gradient;

	}

}

void parameterized_interpolate_test(

)
{
	std::vector<affix_base::data::ptr<element>> l_elements;
	std::vector<affix_base::data::ptr<state_gradient_pair>> l_parameters;

	std::vector<state_gradient_pair> l_x = { {1}, {20}, {30} };

	parameterized_interpolate l_parameterized_interpolate(l_elements, l_parameters, { &l_x[0], &l_x[1], &l_x[2] });

	for (int i = 0; i < l_parameters.size(); i++)
		l_parameters[i]->m_state = (double)i * 0.01;

	for (int epoch = 0; epoch < 1000000; epoch++)
	{
		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		l_parameterized_interpolate.m_y->m_gradient = l_parameterized_interpolate.m_y->m_state - 20;

		if (epoch % 1 == 0)
			std::cout << l_parameterized_interpolate.m_y->m_gradient << std::endl;

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		for (int i = 0; i < l_parameters.size(); i++)
			l_parameters[i]->m_state -= 0.02 * l_parameters[i]->m_gradient;

	}

}

void tnn_test(

)
{
	std::vector<affix_base::data::ptr<element>> l_elements;
	std::vector<affix_base::data::ptr<state_gradient_pair>> l_parameters;

	state_gradient_pair_vector l_x(2);

	tnn l_tnn(l_elements, l_parameters, l_x.pointers(),
		{
			tnn::layer_info(5, neuron_tanh(l_elements, l_parameters)),
			tnn::layer_info(1, neuron_sigmoid(l_elements, l_parameters))
		}
	);

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(25);

	for (int i = 0; i < l_parameters.size(); i++)
	{
		l_parameters[i]->m_state = l_urd(l_dre);
	}

	auto l_cycle = [&](const std::vector<state_gradient_pair>& a_x, const std::vector<state_gradient_pair>& a_y)
	{
		for (int i = 0; i < l_x.size(); i++)
			l_x[i].m_state = a_x[i].m_state;

		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		double l_cost = mean_squared_error(l_tnn.m_y, a_y);
			
		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

		return l_cost;

	};

	const int CHECKPOINT = 100000;

	std::vector<state_gradient_pair_vector> l_ts_x = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1}
	};

	std::vector<state_gradient_pair_vector> l_ts_y = {
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
	std::vector<affix_base::data::ptr<element>> l_elements;
	std::vector<affix_base::data::ptr<state_gradient_pair>> l_parameters;

	state_gradient_pair_vector l_x(1);

	tnn l_tnn(l_elements, l_parameters, l_x.pointers(),
		{
			tnn::layer_info(20, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(1, neuron_leaky_relu(l_elements, l_parameters))
		});

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(25);

	for (int i = 0; i < l_parameters.size(); i++)
	{
		l_parameters[i]->m_state = l_urd(l_dre);
	}

	auto l_cycle = [&](const std::vector<double>& a_x, const std::vector<double>& a_y)
	{
		for (int i = 0; i < l_x.size(); i++)
			l_x[i].m_state = a_x[i];

		for (int i = 0; i < l_elements.size(); i++)
			l_elements[i]->fwd();

		double l_cost = 0;

		for (int i = 0; i < l_tnn.m_y.size(); i++)
		{
			l_tnn.m_y[i]->m_gradient = l_tnn.m_y[i]->m_state - a_y[i];
			l_cost += abs(l_tnn.m_y[i]->m_gradient);
		}

		for (int i = l_elements.size() - 1; i >= 0; i--)
			l_elements[i]->bwd();

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
	std::vector<affix_base::data::ptr<element>> l_elements;

	affix_base::data::ptr<branch> l_branch(new branch(l_elements, true));

	state_gradient_pair l_x_0 = { 1.5 };
	state_gradient_pair l_x_1 = { 2 };

	affix_base::data::ptr<multiply> l_multiply(new multiply(l_branch->elements(), &l_x_0, &l_x_1));

	for (int i = 0; i < l_elements.size(); i++)
		l_elements[i]->fwd();



}

void lstm_test(

)
{
	std::vector<affix_base::data::ptr<element>> l_elements;
	std::vector<affix_base::data::ptr<state_gradient_pair>> l_parameters;

	const size_t l_lstm_y_units = 1;
	const size_t l_tnn_h0_units = 1;
	const size_t l_tnn_y_units = 1;

	auto l_x = state_gradient_pair_matrix(4, 2);

	lstm l_lstm_0(l_elements, l_parameters, l_x.pointers(), l_lstm_y_units);

	tnn l_tnn_0(l_elements, l_parameters, l_lstm_0.m_y[0],
		{
			tnn::layer_info(l_lstm_y_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_h0_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_y_units, neuron_sigmoid(l_elements, l_parameters))
		});
	tnn l_tnn_1(l_elements, l_parameters, l_lstm_0.m_y[1],
		{
			tnn::layer_info(l_lstm_y_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_h0_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_y_units, neuron_sigmoid(l_elements, l_parameters))
		});
	tnn l_tnn_2(l_elements, l_parameters, l_lstm_0.m_y[2],
		{
			tnn::layer_info(l_lstm_y_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_h0_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_y_units, neuron_sigmoid(l_elements, l_parameters))
		});
	tnn l_tnn_3(l_elements, l_parameters, l_lstm_0.m_y[3],
		{
			tnn::layer_info(l_lstm_y_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_h0_units, neuron_leaky_relu(l_elements, l_parameters)),
			tnn::layer_info(l_tnn_y_units, neuron_sigmoid(l_elements, l_parameters))
		});

	std::vector<std::vector<state_gradient_pair*>> l_y = { l_tnn_0.m_y, l_tnn_1.m_y, l_tnn_2.m_y, l_tnn_3.m_y };

	std::uniform_real_distribution<double> l_urd(-1, 1);
	std::default_random_engine l_dre(27);

	for (auto& l_parameter : l_parameters)
	{
		l_parameter->m_state = l_urd(l_dre);
	}

	std::vector<state_gradient_pair_matrix> l_training_set_xs =
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

	std::vector<state_gradient_pair_matrix> l_training_set_ys =
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

	for (int epoch = 0; true; epoch++)
	{
		double l_cost = 0;

		for (int i = 0; i < l_training_set_xs.size(); i++)
		{
			l_x.set_state(l_training_set_xs[i]);
			
			// Carry forward
			for (int j = 0; j < l_elements.size(); j++)
				l_elements[j]->fwd();

			// Signal output
			for (int j = 0; j < l_y.size(); j++)
			{
				for (int k = 0; k < l_y[j].size(); k++)
				{
					l_y[j][k]->m_gradient = l_y[j][k]->m_state - l_training_set_ys[i][j][k].m_state;
					l_cost += abs(l_y[j][k]->m_gradient);
				}
			}

			// Carry backward
			for (int j = l_elements.size() - 1; j >= 0; j--)
				l_elements[j]->bwd();

		}

		for (int i = 0; i < l_parameters.size(); i++)
		{
			l_parameters[i]->m_state -= 0.2 * l_parameters[i]->m_gradient;
			l_parameters[i]->m_gradient = 0;
		}

		if (epoch % 1000 == 0)
			std::cout << l_cost << std::endl;

	}

}

int main(

)
{
	srand(time(0));

	tnn_test();

	return 0;
}
