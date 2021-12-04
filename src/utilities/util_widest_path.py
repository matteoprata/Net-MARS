# Python3 implementation of the approach
import src.constants as co
import networkx as nx
from src.preprocessing.graph_utils import make_existing_edge

# Function to print required path
def printpath(parent, vertex, target, out):
    # global parent
    if (vertex == 0):
        return
    printpath(parent, parent[vertex], target, out)
    out += [vertex]
    return out
    # print(vertex, end="\n" if (vertex == target) else "--")


# Function to return the maximum weight
# in the widest path of the given graph
def widest_path(G, src, target):
    # To keep track of widest distance
    widest = {n: co.epsilon for n in G.nodes}

    # To get the path at the end of the algorithm
    parent = {n: 0 for n in G.nodes}

    # Use of Minimum Priority Queue to keep track minimum
    # widest distance vertex so far in the algorithm
    container = []
    container.append((0, src))
    widest[src] = co.epsilon ** -1
    container = sorted(container)
    while (len(container) > 0):
        temp = container[-1]
        current_src = temp[1]
        del container[-1]

        for neigh in G.neighbors(current_src):
            # Finding the widest distance to the vertex
            # using current_source vertex's widest distance
            # and its widest distance so far

            n1, n2 = make_existing_edge(G, current_src, neigh)
            cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.CAPACITY.value]
            res_cap = G.edges[n1, n2, co.EdgeType.SUPPLY.value][co.ElemAttr.RESIDUAL_CAPACITY.value]
            edge_unit = cap / ((cap - res_cap)+co.epsilon)

            distance = max(widest[neigh],
                           min(widest[current_src], edge_unit))

            # Relaxation of edge and adding into Priority Queue
            if (distance > widest[neigh]):
                # Updating bottle-neck distance
                widest[neigh] = distance

                # To keep track of parent
                parent[neigh] = current_src

                # Adding the relaxed edge in the priority queue
                container.append((distance, neigh))
                container = sorted(container)

    a = printpath(parent, target, target, [])
    # print("max path:", a, "capacity:", widest[target])
    return a #widest[target]


# # Driver code
# if __name__ == '__main__':
#     # Graph representation
#     # graph = [[] for i in range(5)]
#     # no_vertices = 4
#     # # Adding edges to graph
#     #
#     # # Resulting graph
#     # # 1--2
#     # # |  |
#     # # 4--3
#     #
#     # # Note that order in pair is (distance, vertex)
#     # graph[1].append((1, 2))
#     # graph[1].append((2, 4))
#     # graph[2].append((3, 3))
#     # graph[4].append((5, 3))
#
#
#     G = nx.DiGraph()
#     G.add_nodes_from([1, 2, 3, 4])
#     G.add_edges_from([(1, 2), (1, 4), (2, 3), (4, 3)]) #, (2,1), (4,1), (3,2), (3,4)])
#     G.edges[1, 2][co.ElemAttr.CAPACITY.value] = 1
#     G.edges[1, 4][co.ElemAttr.CAPACITY.value] = 2
#     G.edges[4, 3][co.ElemAttr.CAPACITY.value] = 1
#     G.edges[2, 3][co.ElemAttr.CAPACITY.value] = 3
#
#     print(widest_path_problem(G, 1, 3))
#
# # This code is contributed by mohit kumar 29
