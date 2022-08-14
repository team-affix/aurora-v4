#include "affix-base/pch.h"
#include "model.h"
#include "elements.h"
#include "cryptopp/osrng.h"

using namespace aurora;

std::vector<model> model::s_models;

std::default_random_engine model::s_default_random_engine(26);

void model::begin(

)
{
	s_models.push_back(model());
}

model model::end(

)
{
	model l_result = s_models.back();
	s_models.pop_back();
	return l_result;
}

model model::end(
	const double& a_minimum_parameter_state,
	const double& a_maximum_parameter_state
)
{
	model l_result = end();

	std::uniform_real_distribution<double> l_uniform_real_distribution(a_minimum_parameter_state, a_maximum_parameter_state);

	for (auto& l_parameter : l_result.parameters())
	{
		l_parameter->m_state = l_uniform_real_distribution(s_default_random_engine);
	}

	return l_result;

}

model model::end(
	const double& a_minimum_parameter_state,
	const double& a_maximum_parameter_state,
	const std::function<affix_base::data::ptr<optimizer>(affix_base::data::ptr<state_gradient_pair>)>& a_construct_optimizer
)
{
	model l_result = end(a_minimum_parameter_state, a_maximum_parameter_state);

	for (auto& l_parameter : l_result.parameters())
	{
		l_result.m_optimizers.push_back(a_construct_optimizer(l_parameter));
	}

	return l_result;

}

void model::fwd(

)
{
	for (int i = 0; i < m_elements.size(); i++)
		m_elements[i]->fwd();
}

void model::bwd(

)
{
	for (int i = m_elements.size() - 1; i >= 0; i--)
		m_elements[i]->bwd();
}

void model::update(

)
{
	for (auto& l_optimizer : m_optimizers)
		l_optimizer->update();
}
