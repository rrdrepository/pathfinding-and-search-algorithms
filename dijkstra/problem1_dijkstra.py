"""Problem 1 - Dijkstra's Shortest Path Algorithm.

This console program implements Dijkstra's algorithm manually using an
adjacency matrix.  It deliberately does not use library sorting, heaps, or
graph algorithms for the core solution.
"""


INF = float("inf")


def read_integer(prompt, minimum=None, maximum=None):
    """Read one integer and keep asking until it is within the given range."""
    while True:
        try:
            value = int(input(prompt).strip())
            if minimum is not None and value < minimum:
                print(f"Please enter a value of at least {minimum}.")
            elif maximum is not None and value > maximum:
                print(f"Please enter a value no greater than {maximum}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a whole number.")


def read_positive_weight(prompt):
    """Read an edge weight that is valid for Dijkstra's algorithm."""
    while True:
        weight = read_integer(prompt, minimum=1)
        return weight


def vertex_name(vertex):
    """Convert zero-based vertex index to a friendly one-based label."""
    return f"V{vertex + 1}"


def choose_unvisited_vertex(distances, visited):
    """Manually find the unvisited vertex with the smallest tentative distance."""
    smallest_distance = INF
    selected_vertex = -1

    for vertex in range(len(distances)):
        if not visited[vertex] and distances[vertex] < smallest_distance:
            smallest_distance = distances[vertex]
            selected_vertex = vertex

    return selected_vertex

def dijkstra(graph, source):
    """Return shortest distances and predecessors, printing every greedy choice."""
    vertex_count = len(graph)
    distances = [INF] * vertex_count
    predecessors = [-1] * vertex_count
    visited = [False] * vertex_count
    distances[source] = 0

    print("\nGREEDY SELECTION AND RELAXATION STEPS")
    print("-" * 72)

    for step in range(vertex_count):
        current = choose_unvisited_vertex(distances, visited)
        if current == -1:
            print("No remaining reachable vertices. The graph is disconnected from the source.")
            break

        visited[current] = True
        print(
            f"Step {step + 1}: choose {vertex_name(current)} "
            f"because it has the smallest tentative distance ({distances[current]})."
        )

        relaxed_any_edge = False
        for neighbour in range(vertex_count):
            weight = graph[current][neighbour]
            if weight is None or visited[neighbour]:
                continue

            candidate_distance = distances[current] + weight
            if candidate_distance < distances[neighbour]:
                old_distance = distances[neighbour]
                distances[neighbour] = candidate_distance
                predecessors[neighbour] = current
                old_text = "infinity" if old_distance == INF else str(old_distance)
                print(
                    f"  Update {vertex_name(neighbour)}: {old_text} -> "
                    f"{candidate_distance} via {vertex_name(current)}."
                )
                relaxed_any_edge = True
            else:
                print(
                    f"  Keep {vertex_name(neighbour)} at {distances[neighbour]} "
                    f"(route through {vertex_name(current)} would be {candidate_distance})."
                )

        if not relaxed_any_edge:
            print("  No tentative distances were improved in this step.")

    return distances, predecessors


def build_path(predecessors, source, destination):
    """Reconstruct one shortest path without using reverse()."""
    backwards_path = []
    current = destination

    while current != -1:
        backwards_path.append(current)
        if current == source:
            break
        current = predecessors[current]

    if not backwards_path or backwards_path[-1] != source:
        return []

    path = []
    for index in range(len(backwards_path) - 1, -1, -1):
        path.append(backwards_path[index])
    return path


def print_adjacency_matrix(graph):
    """Display the user-entered graph in a simple table."""
    vertex_count = len(graph)
    print("\nGRAPH ADJACENCY MATRIX (dash = no direct edge)")
    print("     ", end="")
    for vertex in range(vertex_count):
        print(f"{vertex_name(vertex):>6}", end="")
    print()

    for row in range(vertex_count):
        print(f"{vertex_name(row):>4} ", end="")
        for column in range(vertex_count):
            value = graph[row][column]
            text = "-" if value is None else str(value)
            print(f"{text:>6}", end="")
        print()


def print_results(distances, predecessors, source):
    """Print a final shortest-path summary for every vertex."""
    print("\nFINAL SHORTEST-PATH SUMMARY")
    print("-" * 72)
    print(f"{'Destination':<14}{'Distance':<12}Path")
    print("-" * 72)

    for destination in range(len(distances)):
        if distances[destination] == INF:
            distance_text = "Unreachable"
            path_text = "No path from source"
        else:
            path = build_path(predecessors, source, destination)
            distance_text = str(distances[destination])
            path_text = " -> ".join(vertex_name(vertex) for vertex in path)

        print(f"{vertex_name(destination):<14}{distance_text:<12}{path_text}")


def read_graph():
    """Read a weighted directed or undirected graph from the user."""
    print("DIJKSTRA'S SHORTEST PATH ALGORITHM")
    print("All edge weights must be positive integers. Vertices are labelled V1, V2, ...\n")

    vertex_count = read_integer("Number of vertices (2-20): ", minimum=2, maximum=20)
    graph_type = read_integer("Graph type: 1 = undirected, 2 = directed: ", minimum=1, maximum=2)
    graph = [[None for _ in range(vertex_count)] for _ in range(vertex_count)]

    maximum_edges = vertex_count * (vertex_count - 1)
    if graph_type == 1:
        maximum_edges //= 2
    edge_count = read_integer(
        f"Number of edges (0-{maximum_edges}): ", minimum=0, maximum=maximum_edges
    )

    for edge_number in range(1, edge_count + 1):
        print(f"\nEdge {edge_number} of {edge_count}")
        start = read_integer(f"  Start vertex (1-{vertex_count}): ", 1, vertex_count) - 1
        end = read_integer(f"  End vertex (1-{vertex_count}): ", 1, vertex_count) - 1
        while end == start:
            print("  Self-loops are not needed for this program.")
            end = read_integer(f"  End vertex (1-{vertex_count}): ", 1, vertex_count) - 1
        weight = read_positive_weight("  Edge weight: ")

        if graph[start][end] is not None:
            print("  This edge already exists; its weight will be replaced.")
        graph[start][end] = weight
        if graph_type == 1:
            graph[end][start] = weight

    source = read_integer(f"\nSource vertex (1-{vertex_count}): ", 1, vertex_count) - 1
    return graph, source


def main():
    graph, source = read_graph()
    print_adjacency_matrix(graph)
    distances, predecessors = dijkstra(graph, source)
    print_results(distances, predecessors, source)
    print("\nProgram finished successfully.")


if __name__ == "__main__":
    main()