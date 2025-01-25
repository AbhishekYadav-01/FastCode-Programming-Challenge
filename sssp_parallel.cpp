#include "solution.hpp"
// #include <omp.h>
#include <cmath>
#include <vector>
#include <queue>
#include <tuple>
#include <map>
#include <mutex>
#include <limits>

typedef std::pair<weight_type, vidType> WN;

class Graph : public BaseGraph {
    eidType* rowptr;
    vidType* col;
    weight_type* weights;
    uint64_t N;
    uint64_t M;
    std::vector<std::mutex> locks; // Mutex for protecting distance updates

public:
    Graph(eidType* rowptr, vidType* col, weight_type* weights, uint64_t N, uint64_t M) :
        rowptr(rowptr), col(col), weights(weights), N(N), M(M), locks(N) {}

    ~Graph() {}

void SSSP(vidType source, weight_type *distances) {
    std::priority_queue<WN, std::vector<WN>, std::greater<WN>> mq;
    std::vector<bool> visited(N, false);  // Track visited vertices
    distances[source] = 0;
    mq.push(std::make_pair(0, source));

    #pragma omp parallel
    {
        while (true) {
            WN curr;
            bool has_next = false;

            #pragma omp critical
            {
                if (!mq.empty()) {
                    curr = mq.top();
                    mq.pop();
                    has_next = true;
                }
            }

            if (!has_next) break;

            auto td = curr.first;  // Current shortest distance
            auto src = curr.second;

            // Skip if already processed by another thread
            if (visited[src]) continue;

            visited[src] = true;

            // Parallelize the relaxation of neighbors
            #pragma omp parallel for
            for (uint64_t i = rowptr[src]; i < rowptr[src + 1]; i++) {
                vidType dst = col[i];
                weight_type wt = weights[i];
                if (td + wt < distances[dst]) {
                    #pragma omp critical
                    {
                        if (td + wt < distances[dst]) {  // Double-check to avoid race conditions
                            distances[dst] = td + wt;
                            mq.push(std::make_pair(distances[dst], dst));
                        }
                    }
                }
            }
        }
    }
}
};

BaseGraph* initialize_graph(eidType* rowptr, vidType* col, weight_type* weights, uint64_t N, uint64_t M) {
    return new Graph(rowptr, col, weights, N, M);
}
