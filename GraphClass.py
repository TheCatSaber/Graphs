from typing import Any, Iterator, Optional
import random
import itertools


Node = int | str
Colour = Optional[int]


class Graph:
    _nodes: list[Node]
    _adj: dict[Node, list[Node]]
    _colours: dict[Node, Colour]
    _saturation_degree_dict: dict[Node, int]
    _ordered_saturation_degrees: list[Node]


    def __init__(self) -> None:
        """A class to store and access properties of graphs.
        
        Also contains other useful functions, including random
        generation and colouring."""
        self._nodes = []
        self._adj = {}
        self._colours = {}

    def __contains__(self, node: Node) -> bool:
        """From networkx.Graph.__contains__"""
        try:
            return node in self._nodes
        except TypeError:
            return False

    def __iter__(self) -> Iterator[Node]:
        """From networkx.Graph.__iter__"""
        return iter(self._nodes)

    def __getitem__(self, node: Node) -> list[Node]:
        """Allows G[node] to return neighbours of node"""
        return self._adj[node]

    def __len__(self) -> int:
        """How many nodes in graph: for len(G)."""
        return len(self._nodes)

    def random_init(self, random_size: int, random_method:str="Erdos-Renyi",
            node_type:str="int",
            alex_mylet_edge_number:int=0,
            erdos_renyi_p:float=0) -> None:
        """Generate a random graph of size random_size using random_method.
        
        random_size: int: size of graph.
        random_method: str: default "Erdos-Renyi". Valid: "Erdos-Renyi",
        "Alex-Mylet". Method by which to create the random graph.
        node_type: str: default "int". Valid "int", "str".
        "int" -> nodes are labelled 1, 2, 3 etc.,
        "str" -> nodes are labelled "A", "B", "C" ... "AA", "AB" etc.
        Other parameters are for the functions should only be used if
        needed.
        """
        match random_method:
            case "Alex-Mylet":
                self._alex_mylet_random(random_size, alex_mylet_edge_number, node_type)
            case "Erdos-Renyi" | _:
                self._erdos_renyi_random(random_size, erdos_renyi_p, node_type)

    def add_node(self, node: Node) -> None:
        if node in self:
            raise ValueError(f"Node {node} already exists")
        self._nodes.append(node)
        self._adj[node] = []
        self._colours[node] = None

    def add_nodes_from_list(self, node_list: list[Node]) -> None:
        for node in node_list:
            self.add_node(node)

    def add_edge(self, node_one: Node, node_two: Node) -> None:
        """Adds nodes if not already in here"""
        if not node_one in self:
            self.add_node(node_one)
        if not node_two in self:
            self.add_node(node_two)
        if node_two not in self._adj[node_one]:
            self._adj[node_one].append(node_two)
        if node_one not in self._adj[node_two]:
            self._adj[node_two].append(node_one)

    def add_edges_from_list(self, edge_list: list[tuple[Node, Node]]) -> None:
        for node_pair in edge_list:
            self.add_edge(node_pair[0], node_pair[1])

    def _string_node_names(self, number: int) -> list[str]:
        n = 1
        counter = 0
        while True:
            for i in itertools.product((chr(i) for i in range(65, 65+26)), repeat=n):
                yield ''.join(i)
                counter += 1
                if counter == number:
                    return
            n += 1

    def _nodes_init(self, size: int, node_type: str) -> None:
        match node_type:
            case "str":
                # Method to get 'A', 'B', 'C', ... 'AA', 'AB', 'AC' ...
                self.add_nodes_from_list([i for i in self._string_node_names(size)])
            case "int" | _:
                self.add_nodes_from_list([i for i in range(1, size+1)])


    def _alex_mylet_random(self, size: int, edge_number: int, node_type: str) -> None:
        self._nodes_init(size, node_type)
        for node in self._nodes:
            for _ in range(edge_number):
                node_choice = random.choice(self._nodes)
                if node_choice != node:
                    self.add_edge(node, node_choice)

    def _compare_nodes(self, node_one: Node, node_two: Node) -> bool:
        if isinstance(node_one, int) and isinstance(node_two, int):
            return node_one < node_two
        elif isinstance(node_one, str) and isinstance(node_two, str):
            return node_one < node_two
        else:
            return isinstance(node_one, int) and isinstance(node_two, str)
    
    def _erdos_renyi_random(self, size: int, p: float, node_type: str) -> None:
        self._nodes_init(size, node_type)
        for node_one in self:
            for node_two in self:
                # This is a deterministic way to compare to between the two nodes
                # So that each edge option is only questioned once
                if self._compare_nodes(node_one, node_two):
                    if p >= random.random():
                        self.add_edge(node_one, node_two)

    def valid_colouring(self) -> bool:
        """Checks if self._colours is a valid colouring
        of the graph made by self._nodes and self._adj."""
        for node in self._nodes:
            for edge_node in self._adj[node]:
                if self._colours[node] == self._colours[edge_node]:
                    return False
        return True

    def _neighbour_colours(self, node: Node) -> set[Colour]:
        return set((self._colours[neighbour] for neighbour in self[node]))
    
    def _colour_node(self, node: Node) -> Colour:
        # neighbour_colours may include None,
        # but this is not an issue as the while loop checks numbers
        # starting from 0 (`None`s are ignored).
        neighbour_colours = self._neighbour_colours(node)
        colour = 0
        while True:
            if colour not in neighbour_colours:
                return colour
            colour += 1

    def greedy_colouring(self, order: list[Node]) -> None:
        if order == None:
            raise TypeError("Order for greedy_colouring cannot be None")
        if len(order) != len(self):
            raise ValueError("Order for greedy_colouring must have the same number of nodes \
                              as the graph")
        for node in order:
            self._colours[node] = self._colour_node(node)

    def random_ordering(self) -> list[Node]:
        nodes = self._nodes[:]
        random.shuffle(nodes)
        return nodes

    def _order_dict_return_list(self, dictionary: dict[Any, Any], reverse:bool=False) -> list[Any]:
        return [key for key, _ in sorted(dictionary.items(),
            key=lambda item: item[1], reverse=reverse)]

    def degree_ordering(self) -> list[Node]:
        vertex_degree_dict = {node: len(self._adj[node]) for node in self}
        ordered_vertex_degress = self._order_dict_return_list(
            vertex_degree_dict, reverse=True)
        return ordered_vertex_degress

    def _make_saturation_degree_dict(self) -> None:
        self._saturation_degree_dict = {node: len(self._neighbour_colours(node))
            for node in self.degree_ordering() if self._colours[node] == None}

    def _order_saturation_degree_dict(self) -> None:
        self._ordered_saturation_degrees = self._order_dict_return_list(
            self._saturation_degree_dict, True)

    def dsatur_colouring(self) -> None:
        self._make_saturation_degree_dict()
        for _ in range(len(self)):
            self._order_saturation_degree_dict()
            node_to_colour = self._ordered_saturation_degrees[0]
            self._colours[node_to_colour] = self._colour_node(node_to_colour)
            # Update saturation_degree_dict: remove coloured node
            self._saturation_degree_dict.pop(node_to_colour)
            # Update neighbours, if not already got a colour,
            # i.e. colour is None
            for neighbour in self[node_to_colour]:
                if self._colours[neighbour] == None:
                    self._saturation_degree_dict[neighbour] = len(self._neighbour_colours(neighbour))
            # Order at start of next iteration