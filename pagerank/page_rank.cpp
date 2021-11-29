#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
//#define OMP_NUM_THREADS		32
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

    int numNodes = num_nodes(g);
    double equal_prob = 1.0 / numNodes;
	double damp_coeff = damping / numNodes;
	double tele_coeff = equal_prob - damp_coeff;
    int* sinks((int*)calloc(numNodes, sizeof(int)));
    int numSinks = 0;
  
    double* new_scores((double*)calloc(numNodes, sizeof(double)));
    if (new_scores == nullptr || sinks == nullptr) {
		printf("\nERROR: Unable to malloc temporary arrays\n");
		abort();
    }
  
    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i) {
		solution[i] = equal_prob;
		if (outgoing_size(g, i) == 0) {
			#pragma omp capture
			{
				int idx = numSinks++;
				sinks[idx] = i;
			}
		}
    }
  
	double curr = 0;
	double prev = 0xDEADBEEF;
	
	while (abs(curr - prev) > convergence) {
		prev = curr;
		curr = 0;
		double curr_tele_add = 0;
		
		#pragma omp parallel for reduction(+:curr_tele_add)
		for (int i = 0; i < numSinks; ++i) {
			curr_tele_add += solution[sinks[i]];
		}
		curr_tele_add *= damp_coeff;
		
		#pragma omp parallel for schedule(guided) reduction(+:curr)
		for (int i = 0; i < numNodes; ++i) {
			// Vertex* points into g.outgoing_edges[]
			double new_score = 0;
			const Vertex* start = incoming_begin(g, i);
			const Vertex* end = incoming_end(g, i);
			for (const Vertex* v = start; v != end; v++) {
				new_score += solution[*v] / outgoing_size(g, *v);
			}
			new_score = (new_score * damping) + tele_coeff + curr_tele_add;
			curr += abs(new_score - solution[i]);
			new_scores[i] = new_score;
		}
		
		#pragma omp parallel for 
		for (int i = 0; i < numNodes; ++i) {
			solution[i] = new_scores[i];
		}
	}
    
	free(sinks);
	free(new_scores);
}  
  
  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */

