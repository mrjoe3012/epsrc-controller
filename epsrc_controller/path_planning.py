from __future__ import annotations
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
from numpy.typing import NDArray
from scipy.stats import norm
from epsrc_controller.utils import vector_angle
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import json

class NodeType:
    CAR_START = 0
    BLUE_CONE = 1
    ORANGE_CONE = 2
    YELLOW_CONE = 3
    LARGE_ORANGE_CONE = 4
    UNKNOWN_CONE = 5
    NUM_NODE_TYPES = 6
    
    def __iter__(self):
        return range(NodeType.NUM_NODE_TYPES)
    
    COLOURS = [
        "green",
        "blue",
        "orange",
        "yellow",
        "orange",
        "purple"
    ]
    
class EdgeDictionary:
    def __init__(self):
        """
        Maps edges which are described by the two vertices that form them as a tuple of two indices to
        a list of other edges described in a similar way.
        The order in which the vertices appear in the tuple doesn't matter.
        """
        self._dict = {}
        self._edges = []
        self._num_unique_edges = 0
        
    def __getitem__(self, edge: tuple[int, int] | int) -> list[tuple[int, int]] | tuple[int, int]:
        """
        Can accept indices in either edge form or integer form. Whenever edges are created by other methods,
        they are automatically assigned and arbitrary index. The integer form of the index corresponds to this.
        Edges are represented internally as pairs of vertex indices. However, depending on the ordering of this tuple,
        there are two distinct representations for each edge. This method checks for this case and returns the correct
        edge information regardless of which ordering is used to index it, provided the edge has an entry internally.
        
        If the integer indexing method is used, this method returns the associated tuple representation for the edge.

        If the tuple indexing method is used, this method returns a list of other edges which are the neighbouring edges
        of the edge in question.

        :param edge: The edge or integer to use as an index.
        :returns: The neighbour list or edge depending on the index type.
        """
        if type(edge) == tuple:
            rev = edge[::-1]
            if edge in self._dict:
                return self._dict[edge]
            elif rev in self._dict:
                return self._dict[rev]
            else:
                self._dict[edge] = (self._num_unique_edges, [])
                self._num_unique_edges += 1
                self._edges.append(edge)
                return self._dict[edge]
        elif type(edge) == int:
            if edge < 0 or edge >= self._num_unique_edges:
                raise IndexError()
            else:
                edge = self._edges[edge]
                return edge
        else:
            raise ValueError()
            
    def __iter__(self):
        return iter(self._dict.items())
        
    def connect(self, edge1: tuple[int, int], edge2: tuple[int, int]) -> None:
        """
        Form an undirected connection from edge1 to edge 2. Add each edge into the other's list
        of neighbours. Doesn't check for duplicates.
        :param edge1: The first edge to connect.
        :param edge2: The second edge to connect.
        """
        l = self[edge1][1]
        l.append(edge2)
        l = self[edge2][1]
        l.append(edge1)
        
    def adjacency(self) -> NDArray:
        """
        Constructs an adjacency matrix for the graph.
        """
        result = np.zeros([self._num_unique_edges for i in range(2)], dtype=np.int32)
        for edge, (edge_idx, neighbours) in self._dict.items():
            neighbour_indices = [self[x][0] for x in neighbours]
            result[edge_idx][neighbour_indices] = 1
        return result
    
class Path:
    def __init__(self, path: list[tuple[int, int]] | None = None):
        """
        A path is represented by a list of edges which are being crossed.
        :param path: The path to initialise with or None to create an empty path.
        """
        if path is None:
            path = []
        self._path = path
    
    def append(self, x: tuple[int, int]) -> None:
        """
        Add a point to the path. This point is represented as the edge which is being crossed.
        :param x: The path point to append represented as an edge between nodes.
        """
        self._path.append(x)

    def __add__(self, other: Path | tuple[int, int]) -> Path:
        """
        Concatenate two paths together.
        :param other: The other path to concatenate to this one.
        :returns: The resulting path.
        """
        result = Path()
        result._path = list(self._path)
        if type(other) == tuple:
            result._path.append(other)
        else:
            result._path.extend(other._path)
        return result
        
    def __iter__(self):
        return iter(self._path)
    
    def __len__(self):
        return len(self._path)
    
    def __getitem__(self, index):
        return self._path[index]
    
class TrackNetwork:
    def __init__(self, points: NDArray, node_types: list[int]):
        """
        Network representation of a track.
        :param points: The coordinates of all nodes on the track except for the car's position.
        Assumed to in the car's frame of reference. Nx2
        :param node_types: A parallel array describing the node types of points. These node types describe
        what colours of cones have been observed.
        """
        # add the car's position to the input data
        points = np.vstack([
            [0.0, 0.0], points
        ])
        self._points = points
        node_types = [NodeType.CAR_START] + node_types
        self._node_types = node_types
        # perform delaunay triangulation and construct a line graph representing drivable points
        # on the track
        self._delaunay = Delaunay(points)
        self._root_edges, self._edge_dict = self._init_edge_dict(self._delaunay)

    def _init_edge_dict(self, delaunay: Delaunay) -> tuple[list[tuple[int, int]], EdgeDictionary]:
        """
        Performs delaunay triangulation and constructs the resultant
        line graph.
        :param delaunay: The scipy.spatial.Delaunay object containing the triangulation.
        :returns: The first element in the tuple is a list of so-called 'root edges'. These are drivable points directly connected
        to the car's starting point. The second element is the dictionary representation of the line-graph, excluding the car's
        start position as a node.
        """
        # construct a set of root edges as to not add duplicate edges
        root_edges, edge_dict = set(), EdgeDictionary()
        # this part gets a bit confusing
        for simplex in delaunay.simplices:
            # a 'simplex' is a triangle represented by three indices pointing to the
            # original node coordinates passed in to the constructor

            # now construct edges from this simplex, these are tuples of indices of vertices. I.e.
            # the two extreme points on the edge.
            edges = [(simplex[i], simplex[(i+1)%3]) for i in range(3)]
            for i in range(3):
                # since we are constructing a line graph, and we are dealing with simplices,
                # each pair of consecutive edges are connected via a vertex
                edge_pairs = [edges[i], edges[(i+1)%3]]
                # now, if either edge in our pair contains the car's starting location, we do not want
                # to add it as a node in our resultant line graph as this makes the path planning confusing

                # however, if only one of the edges contains the car's starting position, then the other edge
                # is considered a 'root edge' as it is a drivable point connected to the car's starting position.
                x = np.array([0 not in edge_pairs[0], 0 not in edge_pairs[1]])
                # this statement handles the case in which neither of the edges contains the car's starting point
                # as a vertex and so it is safe to make a connection between the edges in the network
                if np.all(x):
                    edge_dict.connect(edge_pairs[0], edge_pairs[1])
                # this statement handles the case in which only one of the edges contains the car's starting position
                # and so we don't make a connection but one of the edges should be added to the set of 'root edges'
                elif np.any(x):
                    edge = edge_pairs[np.where(x)[0].item()]
                    root_edges.update([edge])
                # else:
                # this case occurs when both edges contain the car's starting position and so both edges are useless
                # to us
        # now just need to convert the set to a list
        return list(root_edges), edge_dict
     
    def _get_ppf(self, z_score: float) -> float:
        """
        Return the probability point function of some z-score.
        :param z_score: The z-score
        :returns: The corresponding ppf [0,1]
        """
        return 2.0 * norm.cdf(z_score) - 1.0

    def _get_node_ordering(self, vertex_a: NDArray, edge_b: tuple[int, int]) -> tuple[int, int]:
        """
        Returns a tuple containing the indices in b such that the first element is on the left and the
        second element is the one on the right for the path defined by a to b. Accepts the vertex for the
        first edge instead of its index representation as it provides more flexibility when working with
        the first path point, which has an implicit previous vertex of [0,0] (the car's starting point)
        :param a: The vertex of the first path edge.
        :param b: The second path edge.
        :returns: Indices from the second edge such that the first element is the vertex on the left and the second is the
        vertex on the right.
        """
        # calculate vector of path and vectors from vertex to nodes
        vertex_b = self.get_edge_vertex(edge_b)
        cone1 = self._points[edge_b[0], :]
        # calculate angle between path vector and cone vectors
        path_vector = vertex_b - vertex_a
        cone1_vector = cone1 - vertex_b
        angle = vector_angle(path_vector, cone1_vector)
        if angle > 0.0:
            return edge_b
        else:
            return edge_b[::-1]

    def _get_colour_costs(self) -> dict[tuple[int, int], float]:
        """
        Returns the dictionary mapping from a tuple containing colour
        on the left and colour on the right to some number in [0, 1]
        :returns: Dictionary from left, right node types to cost
        """
        t = NodeType
        return {
            (t.BLUE_CONE, t.YELLOW_CONE) : 0.0,
            (t.YELLOW_CONE, t.BLUE_CONE) : 1.0,
            (t.BLUE_CONE, t.BLUE_CONE) : 1.0,
            (t.YELLOW_CONE, t.YELLOW_CONE) : 1.0,
            (t.UNKNOWN_CONE, t.UNKNOWN_CONE) : 0.75,
            (t.BLUE_CONE, t.UNKNOWN_CONE) : 0.5,
            (t.UNKNOWN_CONE, t.BLUE_CONE) : 1.0,
            (t.YELLOW_CONE, t.UNKNOWN_CONE) : 1.0,
            (t.UNKNOWN_CONE, t.YELLOW_CONE) : 0.5,
            (t.LARGE_ORANGE_CONE, t.LARGE_ORANGE_CONE) : 0.5,
            (t.ORANGE_CONE, t.ORANGE_CONE) : 0.5,
            (t.LARGE_ORANGE_CONE, t.UNKNOWN_CONE) : 0.5,
            (t.UNKNOWN_CONE, t.LARGE_ORANGE_CONE) : 0.5,
            (t.ORANGE_CONE, t.UNKNOWN_CONE) : 0.5,
            (t.UNKNOWN_CONE, t.ORANGE_CONE) : 0.5,
            (t.YELLOW_CONE, t.ORANGE_CONE) : 1.0,
            (t.ORANGE_CONE, t.YELLOW_CONE) : 0.5,
            (t.BLUE_CONE, t.ORANGE_CONE) : 0.5,
            (t.ORANGE_CONE, t.BLUE_CONE) : 1.0,
            (t.YELLOW_CONE, t.LARGE_ORANGE_CONE) : 1.0,
            (t.LARGE_ORANGE_CONE, t.YELLOW_CONE) : 0.5,
            (t.BLUE_CONE, t.LARGE_ORANGE_CONE) : 0.5,
            (t.LARGE_ORANGE_CONE, t.BLUE_CONE) : 1.0,
            (t.ORANGE_CONE, t.LARGE_ORANGE_CONE) : 0.5,
            (t.LARGE_ORANGE_CONE, t.ORANGE_CONE) : 0.5,
        }

    def cost(self, path: Path, p=False) -> float:
        """
        Calculates the cost of a path.
        The cost functions is made up of the following:
        
        - Maximum vertex-to-vertex angle change
        - Maximum probability of a vertex being illegal to drive through
        - Standard deviation of vertex-to-vertex distances
        - Standard deviation of vertex edge lengths (distance between cones)

        Each of these values are normalised to [0-1] and summed
        The cost function is based on the one outlined in this paper: https://arxiv.org/abs/1905.05150
        
        :param path: The path to calculate a cost for. Must have at least one vertex inside of it.
        :returns: The cost, a positive floating point number.
        """
        assert len(path) > 0
        _print = lambda *args, **kwargs: print(*args, **kwargs) if p else None
        previous_path_vector = np.array([1.0, 0.0])  # assume heading is 0.0 at start of path
        previous_path_vertex = np.array([0.0, 0.0])  # assume first point on path is car's starting location at origin
        maximum_angle_change = 0.0
        maximum_illegal_crossing_probability = 0.0
        vertex_to_vertex_distances = []
        edge_lengths = []
        colour_costs = self._get_colour_costs()
        
        for edge in path:
            # angle change
            current_path_vertex = self.get_edge_vertex(edge)
            current_path_vector = current_path_vertex - previous_path_vertex
            angle_change = abs(vector_angle(previous_path_vector, current_path_vector))
            if angle_change > maximum_angle_change: maximum_angle_change = angle_change
            # illegal crossing probability
            left_edge_vertex_idx, right_edge_vertex_idx = self._get_node_ordering(previous_path_vertex, edge)
            left_edge_type, right_edge_type = self._node_types[left_edge_vertex_idx], self._node_types[right_edge_vertex_idx]
            left_edge_vertex, right_edge_vertex = self._points[left_edge_vertex_idx, :], self._points[right_edge_vertex_idx]
            illegal_crossing_probability = colour_costs[(left_edge_type, right_edge_type)]
            if illegal_crossing_probability > maximum_illegal_crossing_probability:
                maximum_illegal_crossing_probability = illegal_crossing_probability
            # vertex-to-vertex distance
            vertex_to_vertex_distance = np.linalg.norm(current_path_vector)
            vertex_to_vertex_distances.append(vertex_to_vertex_distance)
            # edge length
            edge_length = np.linalg.norm(left_edge_vertex - right_edge_vertex)
            edge_lengths.append(edge_length)
            # save current vector and vertex
            previous_path_vector = current_path_vector
            previous_path_vertex = current_path_vertex
        
        vertex_to_vertex_std = np.std(vertex_to_vertex_distances)
        edge_length_std = np.std(edge_lengths)

        # normalisation
        norm_angle_change = maximum_angle_change / np.pi
        norm_illegal_prob = maximum_illegal_crossing_probability
        norm_v_to_v = self._get_ppf(vertex_to_vertex_std)
        norm_edge_len = self._get_ppf(edge_length_std)

        cost = np.sum([
            norm_angle_change, norm_illegal_prob, norm_v_to_v, norm_edge_len
        ])
        
        costs_to_print = {
            "angle_change" : norm_angle_change,
            "illegal_crossing_probability" : norm_illegal_prob,
            "vertex_to_vertex_distance" : norm_v_to_v,
            "edge_length" : norm_edge_len,
            "cost" : cost
        }
        _print(json.dumps(costs_to_print, indent=2))

        return cost
        
    def get_edge_vertex(self, edge: tuple[int, int]) -> NDArray:
        """
        For any normal edge, this returns the centrepoint. Any edge involving
        the car's starting point (index 0) will instead return the car's starting
        point.
        :param edge: The edge to get the vertex for.
        :returns: 2x1 coordinate
        """
        if edge[0] == 0 or edge[1] == 0:
            return self._points[0, :]
        else:
            return np.mean(self._points[edge, :], axis=0)

    def plot(self) -> None:
        """
        Uses matplotlib to visualise the state of the track network.
        Cones are shown with their corresponding colours. The car's starting position is
        green. Drivable points are black with black lines connecting them.
        """
        car_start_colour = NodeType.COLOURS[NodeType.CAR_START]
        drivable_point_colour  = "black"
        drivable_line_colour = "black"
        
        car_start = self._points[0, :]
        other_pts = self._points[1:, :]
        
        for edge, (edge_idx, neighbours) in self._edge_dict:
            vertex = np.mean(self._points[edge, :], axis=0)
            vertex = self.get_edge_vertex(edge)
            plt.plot(vertex[0], vertex[1], "o", color=drivable_point_colour)
            plt.plot(self._points[edge, 0], self._points[edge, 1], "-", color="gray", alpha=0.5)
            for neighbour in neighbours:
                n_vertex = np.mean(self._points[neighbour, :], axis=0)
                n_vertex = self.get_edge_vertex(neighbour)
                plt.plot([vertex[0], n_vertex[0]], [vertex[1], n_vertex[1]], "-", color=drivable_line_colour)
            
        plt.plot(car_start[0], car_start[1], "o", color=car_start_colour)
        
        for i, pt in enumerate(other_pts):
            plt.plot(pt[0], pt[1], "o", color=NodeType.COLOURS[self._node_types[i+1]])
            
    def beam_search(self, beam_width: float, max_num_iterations: int) -> list[Path]:
        """
        Performs a beam search of paths starting from the first point (assumed to be the car's start position at 0,0)
        and returns the result.
        :param beam_width: How many of the best paths to keep at each iterations.
        :param max_num_iterations: Maximum number of iterations to run.
        :return: The paths from the beam search.
        """
        paths = [(np.inf, Path([edge])) for edge in self._root_edges]
        iteration = 0
        while iteration < max_num_iterations:
            new_paths = []
            for cost, path in paths:
                last_edge = path[-1]
                last_edge_idx, neighbours = self._edge_dict[last_edge]
                for edge in neighbours:
                    if edge in path or edge[::-1] in path: continue
                    new_path = path + edge
                    new_path_cost = self.cost(new_path)
                    new_paths.append((new_path_cost, new_path))
            new_paths = sorted(new_paths, key=lambda x: x[0])[:beam_width]
            if len(new_paths) == 0:
                print("Warning, terminating beam search due to no new candidates.")
                break
            paths = new_paths
            iteration += 1
        with open("costs.csv", "a") as f:
            for cost, _ in paths:
                f.write(str(cost) + "\n")
        return paths