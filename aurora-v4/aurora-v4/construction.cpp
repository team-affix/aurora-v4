#include "affix-base/pch.h"
#include "construction.h"
#include "elements.h"
#include "cryptopp/osrng.h"

using namespace aurora;

std::vector<element_vector> element_vector::s_element_vectors;

parameter_vector parameter_vector::s_parameter_vector;
size_t parameter_vector::s_next_index(0);
std::default_random_engine parameter_vector::s_default_random_engine(26);

void element_vector::start(

)
{
	s_element_vectors.push_back(element_vector());
}

element_vector element_vector::stop(

)
{
	element_vector l_result = s_element_vectors.back();
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
