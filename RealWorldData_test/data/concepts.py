import networkx as nx

##########################
## cliques
##########################

# cliques with length > 2
def max_cliques(G):
    return sorted([clique for clique in nx.find_cliques(G) if len(clique)>2])

##########################
## cycles
##########################

# cycle basis of the graph
#def cycle_basis(G):
#    return nx.cycle_basis(G, 0)

def cycle_basis(G, max_num=300):
    all_cycles = []
    all_cycles_set = set()
    for node in G.nodes():
        current_cycles = nx.cycle_basis(G, node)
        current_set = [frozenset(c) for c in current_cycles]
        for idx, s in enumerate(current_set):
            if s not in all_cycles_set:
                all_cycles_set.add(s)
                all_cycles.append(current_cycles[idx])

    if max_num is not None:
        return sorted(all_cycles, key=len, reverse=True)[:max_num]
    else:
        return all_cycles

# remove cycle if it forms a clique
def reduced_cycle_basis(G):
    cycle_b = cycle_basis(G)
    max_c = max_cliques(G)
    return [cy for cy in cycle_b for cl in max_c if not set(cy).issubset(cl)]


##########################
## "line" path
## similar to bridges
##########################

# return list of node and corresponding neighbors when the number of neighbors is le than n
def _le_n_neigh_nodes(G, n):
    l_nodes = {}
    for node in nx.nodes(G):
        neighbors = list(nx.neighbors(G, node))
        if len(neighbors) <= n:
            l_nodes[node] = neighbors
    return l_nodes

# construct the path where each node as
def _search_line_rec(l_node, path, l_nodes): # faster implementation possible with dp or check presence
    path.append(l_node)
    if l_node in l_nodes:
        for neighbors in l_nodes[l_node]:
            if neighbors not in path:
                _search_line_rec(neighbors, path, l_nodes)
    return path

# return the "line" paths of the graph, n constrain the "path size"
def line_paths(G):
    line_paths = []
    l_nodes = _le_n_neigh_nodes(G, 2)

    for l_node in list(l_nodes.keys()):
        path = _search_line_rec(l_node, [], l_nodes)
        line_paths.append(path)

    reduced_line_path = list([list(x) for x in set([frozenset(path) for path in line_paths if len(path) > 2])])
    return reduced_line_path


##########################
## Component
##########################

# Generates nodes in each maximal k-edge-connected component in G.
def k_edge_comp(G, max_num=300, k=2):
    return [comp for comp in sorted(map(sorted, nx.k_edge_components(G, k))) if len(comp) > 2][:max_num]

#A k-component is a maximal subgraph of a graph G that has, at least, node connectivity k: we need to remove at least k nodes
#to break it into more components. k-components have an inherent hierarchical structure because they are nested in terms of connectivity:
#a connected graph can contain several 2-components, each of which can contain one or more 3-components, and so forth.
def k_comp(G, max_num=300):
    return [list(comp[0]) for comp in nx.k_components(G).values()][:max_num]


##########################
## Star
##########################

def _star_rec(G, node, visited, layer):
    if layer == 0:
        return
    visited.add(node)
    for neighbors in list(nx.neighbors(G, node)):
        _star_rec(G, neighbors, visited, layer-1)
    return list(visited)

def star(G, n=2, max_num=300):
    stars = []
    for node in nx.nodes(G):
        stars.append(_star_rec(G, node, set(), n+1))
    return stars[:max_num]


##########################
## Star constellation
##########################

def stars_constellation(G, min_degree=1, max_exception=1, max_num=300):
    stars_constellation = []
    for node in nx.nodes(G):
        node_degree = len(list(nx.neighbors(G, node)))
        valid_neig = [neig for neig in list(nx.neighbors(G, node)) if len(list(nx.neighbors(G, neig)))==1]
        const_degree = len(valid_neig)
        const_exception = node_degree - const_degree
        if const_degree >= min_degree and const_exception <= max_exception:
            stars_constellation.append([node] + valid_neig)

        #print(node, node_degree, valid_neig, const_degree, const_exception)
        #print(node, [(neig, len(list(nx.neighbors(G, neig)))) for neig in list(nx.neighbors(G, node)) if len(list(nx.neighbors(G, neig)))==1])
        #print(node, [(neig, len(list(nx.neighbors(G, neig)))) for neig in list(nx.neighbors(G, node)) if len(list(nx.neighbors(G, neig))) > min_degree])
    return stars_constellation[:max_num]


##########################
## max linepath
##########################

def max_lines(G):
    max_degree = 0
    node_degree_l = []
    for node in G.nodes():
        node_degree = G.degree[node]
        node_degree_l.append((node, node_degree))
        if max_degree < node_degree:
            max_degree = node_degree

    max_nodes = [node for (node, node_degree) in node_degree_l if node_degree == max_degree]

    paths = []
    for i, source in enumerate(max_nodes):
        for target in max_nodes[i:]:
            if source != target:
                try:
                    shortest_path = nx.shortest_path(G, source=source, target=target)
                    paths.append(shortest_path)
                except:
                    pass

    if len(paths) == 0:
        return [[]]
    return paths



##########################
## min linepath
##########################

def min_lines(G, min_degree=3):
    node_degree_l = []
    for node in G.nodes():
        node_degree = G.degree[node]
        node_degree_l.append((node, node_degree))

    min_nodes = [node for (node, node_degree) in node_degree_l if node_degree <= min_degree]

    paths = []
    for i, source in enumerate(min_nodes):
        for target in min_nodes[i:]:
            if source != target:
                try:
                    shortest_path = nx.shortest_path(G, source=source, target=target)
                    paths.append(shortest_path)
                except:
                    pass

    if len(paths) == 0:
        return [[]]
    return paths


##########################
## K-core
##########################

def k_core(G):
    k_core_dict = nx.core_number(G)

    result = {}
    for key, value in k_core_dict.items():
        if value not in result:
            result[value] = []
        result[value].append(key)

    result_lists = list(result.values())
    return result_lists

##########################
## degree_centrality
##########################

def degree_centrality(G):
    degree_dict = nx.degree_centrality(G)
    rounded_data = {key: round(value, 3) for key, value in degree_dict.items()}

    result = {}
    for key, value in rounded_data.items():
        if value not in result:
            result[value] = []
        result[value].append(key)

    result_lists = list(result.values())
    return result_lists


##########################
## community modularity
##########################

def comm_modularity(G):
#    comm = nx.community.naive_greedy_modularity_communities(G)
    comm = nx.community.greedy_modularity_communities(G)

    list_comm = [list(s) for s in comm]
    return list_comm




def get_concept_list(concept_str):
    concept_list_dict = {
        "maxcliques":                   [ {"name": "GMC", "fun": max_cliques, "args": []} ],
        "cyclebasis":                   [ {"name": "GCB", "fun": cycle_basis, "args": []} ],
        "maxcliques_cyclebasis":        [ {"name": "GMC", "fun": max_cliques, "args": []},
                                          {"name": "GCB", "fun": cycle_basis, "args": []} ],
        "linepaths":                    [ {"name": "GLP", "fun": line_paths, "args": []} ],
        "k_edge_comp":                  [ {"name": "kecomp", "fun": k_edge_comp, "args": []} ],
        "constell":                     [ {"name": "const", "fun": stars_constellation, "args": []} ],
        "star2":                        [ {"name": "star2", "fun": star, "args": []} ],
        "maxcliques_cyclebasis_star2":  [ {"name": "GCB", "fun": cycle_basis, "args": []},
                                          {"name": "GMC", "fun": max_cliques, "args": []},
                                          {"name": "star2", "fun": star, "args": []} ],
        "maxlines":                     [ {"name": "maxline", "fun": max_lines, "args": []} ],
        "minlines":                     [ {"name": "minline", "fun": min_lines, "args": []} ],
        "k_core":                       [ {"name": "k_core", "fun": k_core, "args": []} ],
        "degree_centrality":            [ {"name": "deg_cent", "fun": degree_centrality, "args": []} ],
        "comm_modul":                   [ {"name": "comm_mod", "fun": comm_modularity, "args": []} ],
        "cycb_maxcliq_star2_minl_maxl": [ {"name": "GCB", "fun": cycle_basis, "args": []},
                                          {"name": "GMC", "fun": max_cliques, "args": []},
                                          {"name": "star2", "fun": star, "args": []},
                                          {"name": "minline", "fun": min_lines, "args": []},
                                          {"name": "maxline", "fun": max_lines, "args": []} ],
    }
    concept_list = concept_list_dict[concept_str]
    if concept_list is None:
        raise ValueError(f"Concept {concept_str} not found")
    return concept_list










