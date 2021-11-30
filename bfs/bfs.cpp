#include <atomic>
#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

#define VERBOSE

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances,
	vertex_set* scratch)
{	

	#ifdef VERBOSE
	printf("Top down step with %d OpenMP iters = front\n", frontier->count);
	#endif
	#pragma omp parallel for schedule(dynamic, 128) if (omp_get_max_threads() > 1)
    for (int i=0; i<frontier->count; i++) {
        int node = frontier->vertices[i];
		vertex_set local = scratch[omp_get_thread_num()];
        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];
			if (distances[outgoing] == NOT_VISITED_MARKER)/* && 
			__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node]+1))*/ {
				distances[outgoing] = distances[node] + 1;
				local.vertices[local.count++] = outgoing;
				#ifdef VERBOSE
				printf("Node %d edge %d neighbor_vert %d added to new frontier with curr distance %d, count %d\n", node, neighbor, outgoing, distances[node] + 1, local.count);
				#endif
			}
        }
		
		/*int localcount;
		#pragma omp atomic capture
		{
			localcount = new_frontier->count;
			new_frontier->count += local.count;
		}
		
		for (int j = 0; j < local.count; ++j)
			new_frontier->vertices[localcount + j] = local.vertices[j];
			
		local.count = 0;*/
    }
	
	for (int i = 0; i < omp_get_max_threads(); ++i) {
		vertex_set local = scratch[i];
		printf("Loading buffer %d, count %d\n", i, local.count);
		for (int j = 0; j < local.count; ++j) {
			new_frontier->vertices[new_frontier->count + j] = local.vertices[j];
		}
		local.count = 0;
	}
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
	#pragma omp parallel for schedule(static, (64 / sizeof(int)))
    for (int i=0; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;
	vertex_set scratch[omp_get_max_threads()];
	for (int i = 0; i < omp_get_max_threads(); ++i) {
		scratch[i].vertices = new int[graph->num_nodes];
		scratch[i].count = 0;
	}

#ifdef VERBOSE
	int iter = 1;
#endif
    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
#ifdef VERBOSE
		printf("Beginning top down step iteration %d:\n", iter);
#endif
        top_down_step(graph, frontier, new_frontier, sol->distances, scratch);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
	
	for (int i = 0; i < omp_get_max_threads(); ++i)
		delete[] scratch[i].vertices;
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
}
