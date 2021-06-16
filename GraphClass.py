from typing import Iterator
import random


Node = int | str
Colour = int

class Graph:
    _nodes: list[Node]
    _adj: dict[Node, list[Node]]
    _colours: dict[Node, Colour]
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

    def random_init(self, random_size: int, random_method:str="Erdos-Renyi", \
            node_type:str="int",
            alex_mylet_edge_number:int=0, \
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

    def add_nodes_from_list(self, node_list: list[Node]) -> None:
        for node in node_list:
            self.add_node(node)

    def add_edge(self, node_one: Node, node_two: Node) -> None:
        """Adds nodes if not already in here"""
        if not node_one in self:
            self.add_node(node_one)
        if not node_two in self:
            self.add_node(node_two)
        self._adj[node_one].append(node_two)
        self._adj[node_two].append(node_one)

    def add_edges_from_list(self, edge_list: list[tuple[Node, Node]]) -> None:
        for node_pair in edge_list:
            self.add_edge(node_pair[0], node_pair[1])

    def _nodes_init(self, size: int, node_type: str) -> None:
        match node_type:
            case "str":
                # TODO: Method to get A, B, C, ... AA, AB, AC etc.
                self.add_nodes_from_list([chr(i) for i in range(65, 65+size)])
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
