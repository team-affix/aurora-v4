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
			T m_global_best_reward;
			tensor<T, I> m_global_best_position;

		public:
			particle_swarm_optimizer(
				tensor<T, PARTICLE_COUNT, I>& a_positions,
				const T& a_w,
				const T& a_c1,
				const T& a_c2
			) :
                m_positions(a_positions),
                m_local_best_positions(constant<T, PARTICLE_COUNT, I>()),
                m_velocities(constant<T, PARTICLE_COUNT, I>()),
                m_local_best_rewards(constant<T, PARTICLE_COUNT>()),
				m_w(a_w),
				m_c1(a_c1),
				m_c2(a_c2),
                m_global_best_reward(-INFINITY),
                m_global_best_position(constant<T, I>())
			{

			}

			void update(
				const tensor<T, PARTICLE_COUNT>& a_particle_rewards
			)
			{
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

	}

}

#endif
