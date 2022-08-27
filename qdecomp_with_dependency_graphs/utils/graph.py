import itertools
from typing import Callable, Any, List, Dict, Tuple, Iterable, Union

import networkx as nx
from queue import Queue, deque

from spacy import displacy
from html import escape


def has_cycle(graph: nx.DiGraph):
    try:
        nx.find_cycle(graph, orientation='original')
        return True
    except:
        return False


def get_graph_levels(graph: nx.DiGraph):
    """
    Find graph level for each node
    level[node] := 0 if the node has no successors
    level[node] := max[over successors s](level[s])+1
    :param graph: directed graph with no cycles
    :return: (nodes_level, levels) tuple where:
        nodes_level: dictionary of <node_id>:<level:int>
        levels: dictionary of <level:int>:[<node_id>]
    """
    updated_nodes = Queue()

    # first layer
    leafs = [n_id for n_id in graph.nodes if not any(graph.successors(n_id))]
    nodes_levels = {n_id: 0 for n_id in leafs}
    updated_nodes.queue = deque(leafs)

    # update predecessors
    while not updated_nodes.empty():
        n_id = updated_nodes.get()
        low_bound = nodes_levels[n_id] + 1
        if low_bound > graph.number_of_nodes():
            raise ValueError("Cyclic graphs are not allowed")
        for s_id in graph.predecessors(n_id):
            if nodes_levels.get(s_id, -1) < low_bound:
                nodes_levels[s_id] = low_bound
                updated_nodes.put(s_id)
    levels = {}
    for n_id, l in nodes_levels.items():
        levels[l] = levels.get(l, []) + [n_id]

    return nodes_levels, levels


def reorder_by_level(graph: Union[nx.DiGraph, nx.MultiDiGraph], key: Callable[[int, dict], Any], update_node: Callable[[dict, Dict[int, int]], None]):
    """
    Merges identical nodes, so use with DiGraph only if the edges has no attributes
    :param graph:
    :param key:
    :param update_node:
    :return:
    """
    _, levels = get_graph_levels(graph)

    # order by levels
    old_to_new_id_map = {}
    next_node_id = 1
    for level in sorted(levels.keys()):
        nodes_by_key = {}
        for n_id in levels[level]:
            # for consistency
            n = graph.nodes[n_id]
            update_node(n, old_to_new_id_map)
            n_key = key(n_id, n)
            if n_key not in nodes_by_key:
                nodes_by_key[n_key] = n_id
            else:
                # duplicate label
                # safe merge since previous levels are fixed
                exists_node_id = nodes_by_key[n_key]
                # successors - should be the same due to label
                predecessors_edges = graph.in_edges(n_id, data=True)
                for p_id, _, attr in predecessors_edges:
                    graph.add_edge(p_id, exists_node_id, **attr)
                    update_node(graph.nodes[p_id], {n_id: exists_node_id})
                graph.remove_node(n_id)
        nodes_order = sorted(nodes_by_key.keys())
        for n_key in nodes_order:
            n_id = nodes_by_key[n_key]
            old_to_new_id_map[n_id] = next_node_id
            next_node_id += 1

    # update labels
    # double mapping since new and old labels are overlap
    nx.relabel.relabel_nodes(graph, {k:str(v) for k,v in old_to_new_id_map.items()}, copy=False)
    nx.relabel.relabel_nodes(graph, {str(v):v for v in old_to_new_id_map.values()}, copy=False)
    # note: nodes are updated


def render_digraph_svg(graph:nx.MultiDiGraph,
                       words_text_selector: Callable[[int, dict], Any] = None,
                       words_tag_selector: Callable[[int, dict], Any] = None,
                       arc_label_selector: Callable[[dict], Any] = None) -> Any:
    nodes_ids = sorted(graph)
    node_id_to_indexes = {n_id: i for i, n_id in enumerate(nodes_ids)}
    words = [(words_text_selector and str(words_text_selector(n_id, graph.nodes(data=True)[n_id])),
              words_tag_selector and str(words_tag_selector(n_id, graph.nodes(data=True)[n_id])))
             for n_id in nodes_ids]
    arcs = [(node_id_to_indexes[x],node_id_to_indexes[y],
             arc_label_selector and str(arc_label_selector(data)))
            for x,y,data in graph.edges(data=True)]
    return render_dependencies_graph_svg(words, arcs)


def render_dependencies_graph_svg(
        words: Iterable[Tuple[str, str]],  arcs: Iterable[Tuple[int, int, str]]
) -> Any:
    words = [{'text': escape(x or ''), 'tag': escape(y or '')} for x, y in words]
    arcs = [{'label': escape(label or ''),
             'start': min(u, v), 'end': max(u, v), 'dir': 'right' if u < v else 'left'}
            for u, v, label in arcs]

    # fix None
    for x in words+arcs:
        for k in x:
            if x[k] is None:
                x[k] = ''

    # merge arcs labels
    merged_arcs = []
    for k, g in itertools.groupby(arcs, key=lambda x:(x['start'],x['end'])):
        s, e = k
        labels, dirs = zip(*[(x['label'], x['dir']) for x in g])
        for d in set(dirs):
            merged_arcs.append({'label': '', 'start': s, 'end': e, 'dir': d})
        merged_arcs[-1]['label'] = ','.join(labels)

    svg = displacy.render({'arcs': merged_arcs, 'words': words}, manual=True, style='dep')
    return svg