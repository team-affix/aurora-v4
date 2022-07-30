#include "affix-base/pch.h"
#include "model.h"
#include "elements.h"

using namespace aurora;

std::vector<model> model::s_models;

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
