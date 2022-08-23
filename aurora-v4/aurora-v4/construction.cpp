#include "affix-base/pch.h"
#include "construction.h"
#include "cryptopp/osrng.h"

using namespace aurora;

std::vector<element_vector> element_vector::s_element_vectors;

std::vector<parameter_vector> parameter_vector::s_parameter_vectors;

void element_vector::start(

)
{
	s_element_vectors.push_back(element_vector());
}

element_vector element_vector::stop(

)
{
	element_vector l_result = current_element_vector();
	s_element_vectors.pop_back();
	return l_result;
}

void element_vector::fwd(

)
{
	for (int i = 0; i < size(); i++)
		at(i)->fwd();
}

void element_vector::bwd(

)
{
	for (int i = size() - 1; i >= 0; i--)
		at(i)->bwd();
}
