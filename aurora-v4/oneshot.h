#ifndef ONESHOT_H
#define ONESHOT_H

#include <vector>
#include <random>
#include <assert.h>
#include <stdexcept>
#include "fundamentals.h"

namespace aurora
{
	namespace oneshot
	{
        template<typename T, size_t PARTICLE_COUNT, size_t I>
		class particle_swarm_optimizer
		{
		private:
            static std::uniform_real_distribution<double> s_urd;
        
        private:
			tensor<T, PARTICLE_COUNT, I>& m_positions;
            tensor<T, PARTICLE_COUNT, I>  m_local_best_positions;
            tensor<T, PARTICLE_COUNT, I>  m_velocities;
            tensor<T, PARTICLE_COUNT>     m_local_best_rewards;

			T m_w;
			T m_c1;
			T m_c2;
            T m_memory;
			T m_global_best_reward;
			tensor<T, I> m_global_best_position;

		public:
			particle_swarm_optimizer(
				tensor<T, PARTICLE_COUNT, I>& a_positions,
				const T& a_w,
				const T& a_c1,
				const T& a_c2,
                const T& a_memory = 1.0
			) :
                m_positions(a_positions),
                m_local_best_positions(constant<T, PARTICLE_COUNT, I>()),
                m_velocities(constant<T, PARTICLE_COUNT, I>()),
                m_local_best_rewards(constant<T, PARTICLE_COUNT>()),
				m_w(a_w),
				m_c1(a_c1),
				m_c2(a_c2),
                m_memory(a_memory),
                m_global_best_reward(-INFINITY),
                m_global_best_position(constant<T, I>())
			{

			}

			void update(
				const tensor<T, PARTICLE_COUNT>& a_particle_rewards
			)
			{
                // This adds a memory decay to the global best reward, to
                // assist with non-static or random environments.
                m_global_best_reward *= m_memory;
                
				// Get the global best position if it has improved
				for (int i = 0; i < PARTICLE_COUNT; i++)
				{
					if (a_particle_rewards[i] > m_global_best_reward)
					{
						m_global_best_reward = a_particle_rewards[i];
						m_global_best_position = m_positions[i];
					}
				}

				// Update all particle positions
				for (int i = 0; i < PARTICLE_COUNT; i++)
				{
                    update(
                        m_positions[i],
                        m_local_best_positions[i],
                        m_velocities[i],
                        m_local_best_rewards[i],
                        a_particle_rewards[i]
                    );
				}

			}

			T global_best_reward(

			)
			{
				return m_global_best_reward;
			}

			tensor<T, I> global_best_position(

			)
			{
				return m_global_best_position;
			}

        private:
			void update(
                tensor<T, I>& a_position,
                tensor<T, I>& a_local_best_position,
                tensor<T, I>& a_velocity,
                T& a_local_best_reward,
				const T& a_reward
			)
			{
				if (a_reward > a_local_best_reward)
				{
					a_local_best_position = a_position;
					a_local_best_reward = a_reward;
				}

				tensor<T, I> l_weighted_particle_velocity = multiply(a_velocity, m_w);
				tensor<T, I> l_cognitive_term = multiply(multiply(subtract(a_local_best_position, a_position), m_c1), s_urd(i_default_random_engine));
				tensor<T, I> l_social_term = multiply(multiply(subtract(m_global_best_position, a_position), m_c2), s_urd(i_default_random_engine));
				a_velocity = add(add(l_weighted_particle_velocity, l_cognitive_term), l_social_term);
				a_position = add(a_position, a_velocity);

			}

		};

        template<typename T, size_t PARTICLE_COUNT, size_t I>
        std::uniform_real_distribution<double> particle_swarm_optimizer<T, PARTICLE_COUNT, I>::s_urd(0, 1);

        template<size_t PARTICLE_COUNT, size_t I>
        class icpso
        {
		private:
            static std::uniform_real_distribution<double> s_urd;
        
        private:
			tensor<std::vector<double>, PARTICLE_COUNT, I> m_positions;
            tensor<size_t, PARTICLE_COUNT, I>              m_candidate_solutions;
            tensor<std::vector<double>, PARTICLE_COUNT, I> m_local_best_positions;
            tensor<std::vector<double>, PARTICLE_COUNT, I> m_velocities;
            tensor<double, PARTICLE_COUNT>                 m_local_best_rewards;

			double                         m_w;
			double                         m_c1;
			double                         m_c2;
            double                         m_epsilon;
            double                         m_epsilon_compliment;
			double                         m_global_best_reward;
			tensor<std::vector<double>, I> m_global_best_position;
            tensor<size_t, I>              m_global_best_solution;

		public:
			icpso(
                const tensor<size_t, I>& a_distribution_sizes,
				const double& a_w,
				const double& a_c1,
				const double& a_c2,
                const double& a_epsilon
			) :
                m_local_best_rewards(constant<double, PARTICLE_COUNT>()),
				m_w(a_w),
				m_c1(a_c1),
				m_c2(a_c2),
                m_epsilon(a_epsilon),
                m_epsilon_compliment(1.0 - a_epsilon),
                m_global_best_reward(-INFINITY)
			{
                assert(a_w > 0 && a_w < 1);
                assert(a_c1 > 0 && a_c1 < 1);
                assert(a_c2 > 0 && a_c2 < 1);
                assert(a_epsilon > 0 && a_epsilon < 1);
                
                for (int i = 0; i < PARTICLE_COUNT; i++)
                    for (int j = 0; j < I; j++)
                    {
                        // Create distributions. (Not initializing values)
                        m_positions[i][j] = std::vector<double>(a_distribution_sizes[j]);
                        m_local_best_positions[i][j] = std::vector<double>(a_distribution_sizes[j]);
                        m_velocities[i][j] = std::vector<double>(a_distribution_sizes[j]);

                        // INITIALIZE THE POSITION DISTRIBUTIONS RANDOMLY
                        for (int k = 0; k < a_distribution_sizes[j]; k++)
                            m_positions[i][j][k] = s_urd(i_default_random_engine);

                        // Clip and normalize the distribution for this variable for this particle.
                        clip_and_normalize(m_positions[i][j]);

                    }

                // Initialize the global best position.
                for (int i = 0; i < I; i++)
                    m_global_best_position[i] = std::vector<double>(a_distribution_sizes[i]);

			}

            const tensor<size_t, PARTICLE_COUNT, I>& candidate_solutions(

            )
            {
                for (int i = 0; i < PARTICLE_COUNT; i++)
                    for (int j = 0; j < I; j++)
                        m_candidate_solutions[i][j] = sample(m_positions[i][j]);

                return m_candidate_solutions;
                
            }

			void update(
				const tensor<double, PARTICLE_COUNT>& a_particle_rewards
			)
			{
                // Get the maximum immediate reward
                auto l_max_reward = std::max_element(a_particle_rewards.begin(), a_particle_rewards.end());
                size_t l_max_reward_index = l_max_reward - a_particle_rewards.begin();

                // Update global best position and reward if a new best exists
                if (*l_max_reward > m_global_best_reward)
                {
                    update_best_position(
                        m_global_best_position,
                        m_positions[l_max_reward_index],
                        m_candidate_solutions[l_max_reward_index]
                    );
                    m_global_best_reward = *l_max_reward;
                    m_global_best_solution = m_candidate_solutions[l_max_reward_index];
                }

				// Update all particle positions
				for (int i = 0; i < PARTICLE_COUNT; i++)
				{
                    update(
                        m_positions[i],
                        m_candidate_solutions[i],
                        m_local_best_positions[i],
                        m_velocities[i],
                        m_local_best_rewards[i],
                        a_particle_rewards[i]
                    );
				}

			}

			double global_best_reward(

			)
			{
				return m_global_best_reward;
			}

            const tensor<size_t, I>& global_best_solution(

            )
            {
                return m_global_best_solution;
            }

        private:
            void update_best_position(
                tensor<std::vector<double>, I>& a_old_best_position,
                const tensor<std::vector<double>, I>& a_new_best_position,
                const tensor<size_t, I>& a_candidate_solution
            )
            {
                for (int i = 0; i < I; i++)
                    update_best_distribution(a_old_best_position[i], a_new_best_position[i], a_candidate_solution[i]);
            }

            void update_best_distribution(
                std::vector<double>& a_old_best_distribution,
                const std::vector<double>& a_new_best_distribution,
                const size_t& a_selected_variable_index
            )
            {
                ////////////////////////
                // FOR EACH PROBABILITY IN THE DISTRIBUTION WHICH
                // WAS NOT SAMPLED, DECREASE ITS VALUE (using decay so it will never be negative).
                // AND WHATEVER PROBABILITY VALUE IS LOST FOR EACH PROBABILITY,
                // INCREASE THE PROBABILITY OF CHOOSING THE SELECTED VARIABLE AGAIN.
                ////////////////////////

                a_old_best_distribution[a_selected_variable_index] = a_new_best_distribution[a_selected_variable_index];

                for (int i = 0; i < a_old_best_distribution.size(); i++)
                {
                    if (i == a_selected_variable_index)
                        continue;
                    
                    a_old_best_distribution[i] = m_epsilon * a_new_best_distribution[i];
                    a_old_best_distribution[a_selected_variable_index] += m_epsilon_compliment * a_new_best_distribution[i];
                    
                }
                
            }
        
			void update(
                tensor<std::vector<double>, I>& a_position,
                const tensor<size_t, I>&        a_candidate_solution,
                tensor<std::vector<double>, I>& a_local_best_position,
                tensor<std::vector<double>, I>& a_velocity,
                double&                         a_local_best_reward,
				const double&                   a_reward
			)
			{
				if (a_reward > a_local_best_reward)
				{
                    update_best_position(a_local_best_position, a_position, a_candidate_solution);
					a_local_best_reward = a_reward;
				}

                // Generate two random values
                double l_rand_0 = s_urd(i_default_random_engine);
                double l_rand_1 = s_urd(i_default_random_engine);

                ////////////////////////////////////
                // UPDATE THE VELOCITY AND POSITION VECTORS
                ////////////////////////////////////
                for (int i = 0; i < PARTICLE_COUNT; i++)
                    for (int j = 0; j < I; j++)
                        for (int k = 0; k < m_positions[i][j].size(); k++)
                        {
                            double& l_velocity = m_velocities[i][j][k];
                            double& l_position = m_positions[i][j][k];
                            const double& l_local_best_position = m_local_best_positions[i][j][k];
                            const double& l_global_best_position = m_global_best_position[j][k];

                            // UPDATE VELOCOTY VALUE
                            l_velocity = 
                                m_w * l_velocity + 
                                l_rand_0 * (l_local_best_position - l_position) +
                                l_rand_1 * (l_global_best_position - l_position);

                            // UPDATE POSITION VALUE
                            l_position += l_velocity;
                            
                        }
                
			}

            static size_t sample(
                const std::vector<double>& a_distribution
            )
            {
                double l_remainder = s_urd(i_default_random_engine);

                size_t i = 0;

                for (; i < a_distribution.size() && l_remainder > 0; l_remainder -= a_distribution[i], i++);
                
                return i - 1;
                
            }

            static void clip_and_normalize(
                std::vector<double>& a_distribution
            )
            {
                double l_normalization_denominator = 0;

                for (int i = 0; i < a_distribution.size(); i++)
                {
                    // Clip the value in the distribution.
                    a_distribution[i] = std::min(std::max(a_distribution[i], 0.0), 1.0);

                    l_normalization_denominator += a_distribution[i];
                    
                }

                for (int i = 0; i < a_distribution.size(); i++)
                    a_distribution[i] /= l_normalization_denominator;
                
            }

        };

        template<size_t PARTICLE_COUNT, size_t I>
        std::uniform_real_distribution<double> icpso<PARTICLE_COUNT, I>::s_urd(0, 1);

	}

}

#endif
