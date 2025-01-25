#include "solution.hpp"
#include <cmath>
#include <vector>
#include <limits>
#include <omp.h>

class Graph : public BaseGraph {
    eidType* rowptr;  // CSR row pointer
    vidType* col;     // CSR column indices
    uint64_t N;       // Number of vertices
    uint64_t M;       // Number of edges

public:
    Graph(eidType* rowptr, vidType* col, uint64_t N, uint64_t M) :
        rowptr(rowptr), col(col), N(N), M(M) {}

    ~Graph() {
    }

    void BFS(vidType source, weight_type* distances) {
        // Initialize distances to infinity
        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i++) {
            distances[i] = std::numeric_limits<weight_type>::max();
        }
        distances[source] = 0; // Distance to source is 0

        // Initialize the first frontier with the source node
        std::vector<vidType> this_frontier;
        this_frontier.push_back(source);

        // Use thread-local frontiers to reduce mutex contention
        int max_threads = omp_get_max_threads();
        std::vector<std::vector<vidType>> thread_local_frontiers(max_threads);

        while (!this_frontier.empty()) {
            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                thread_local_frontiers[thread_id].clear(); // Clear thread-local frontier

                // Parallel processing of the current frontier
                #pragma omp for schedule(dynamic, 64) // Dynamic scheduling for load balancing
                for (size_t idx = 0; idx < this_frontier.size(); idx++) {
                    vidType src = this_frontier[idx];
                    for (eidType i = rowptr[src]; i < rowptr[src + 1]; i++) {
                        vidType dst = col[i];
                        weight_type old_distance = distances[dst];
                        weight_type new_distance = distances[src] + 1;

                        // Atomically update the distance if the new one is shorter
                        if (new_distance < old_distance) {
                            if (__sync_bool_compare_and_swap(&distances[dst], old_distance, new_distance)) {
                                // Add to the thread-local frontier
                                thread_local_frontiers[thread_id].push_back(dst);
                            }
                        }
                    }
                }
            }

            // Combine thread-local frontiers into a global next_frontier
            std::vector<vidType> next_frontier;
            for (int t = 0; t < max_threads; t++) {
                next_frontier.insert(next_frontier.end(),
                                     thread_local_frontiers[t].begin(),
                                     thread_local_frontiers[t].end());
            }

            // Swap frontiers
            this_frontier.swap(next_frontier);
        }
    }
};

BaseGraph* initialize_graph(eidType* rowptr, vidType* col, uint64_t N, uint64_t M) {
    return new Graph(rowptr, col, N, M);
}
