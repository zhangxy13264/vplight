import xml.etree.ElementTree as ET
import networkx as nx
import numpy as np
from common.registry import Registry
from common.utils import build_index_intersection_map_sumo


class IntersectionFlowGenerator():
    def __init__(self, world, rank, fns=['lane_vechile_info']):
        self.world = world
        self.fns = fns
        self.world.subscribe(fns)
        self.num_intersections = len(self.world.intersections)

        param = Registry.mapping['world_mapping']['setting'].param
        roadnetFile = param['dir'] + param['roadnetFile']
        index_intersection_map = build_index_intersection_map_sumo(roadnetFile)
        self.cur_id = index_intersection_map['node_idx2id'][rank]
        valid_node_id = set(index_intersection_map['node_id2idx'].keys())
        
        self.lanes = self._get_inter_lanes(self.cur_id)
        self.ob_length = len(self.lanes)
        if self.num_intersections > 1:
            G, edges = self._get_G_and_edges(roadnetFile, valid_node_id)
            self.adj_ids = [neighbor for neighbor in G.neighbors(self.cur_id)]
            self.cur_edges = self._get_cur_edges(edges)
            self.I = self.world.id2intersection[self.cur_id]
            self.adj_inter2from_lane, self.adj_inter2target_road = self._get_adj_inter2info()
            self.roadlinks = self._get_roadlinks()
        
    def _get_G_and_edges(self, roadnetFile, valid_node_id):
        roadnet_tree = ET.parse(roadnetFile)
        roadnet_root = roadnet_tree.getroot()
        edges = {}
        for edge in roadnet_root.findall('edge'):
            if 'from' in edge.attrib and 'to' in edge.attrib:
                edge_id = edge.attrib['id']
                from_node = edge.attrib['from']
                to_node = edge.attrib['to']
                edges[edge_id] = (from_node, to_node)
        G = nx.DiGraph()
        all_nodes = set()
        for edge in edges.values():
            if edge[0] not in valid_node_id or edge[1] not in valid_node_id:
                continue
            all_nodes.add(edge[0])
            all_nodes.add(edge[1])
        G.add_nodes_from(all_nodes)
        for node1, node2 in edges.values():
            if node1 not in valid_node_id or node2 not in valid_node_id:
                continue
            G.add_edge(node1, node2)
        return G, edges
    
    def _get_cur_edges(self, edges):
        cur_edges = {}
        for adj_id in self.adj_ids:
            tmp = {(start, end): key for key, (start, end) in edges.items() if start == adj_id and end == self.cur_id}
            tmp2 = {(start, end): key for key, (start, end) in edges.items() if start == self.cur_id and end == adj_id}
            cur_edges.update(tmp)
            cur_edges.update(tmp2)
        return cur_edges
        
    def _get_inter_lanes(self, inter_id):
        lanes = []
        I = self.world.id2intersection[inter_id]
        for r in I.in_roads:
            if not self.world.RIGHT:
                tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]), reverse=True)
            else:
                tmp = sorted(I.road_lane_mapping[r], key=lambda ob: int(ob[-1]))
            lanes.append(tmp)
        return [lane for road in lanes for lane in road]
    
    def _get_adj_inter2info(self):
        adj_inter2from_lane = {}
        adj_inter2target_road = {}
        for adj_id in self.adj_ids:
            lane_to_tmp = []
            target_road = self.cur_edges[(adj_id, self.cur_id)]
            adj_inter2target_road[adj_id] = target_road
            adj_inter_obj = self.world.id2intersection[adj_id]
            for lanelink in adj_inter_obj.lanelinks:
                lane_from, lane_to, _ = lanelink[0]
                if lane_to in self.I.road_lane_mapping[target_road]:
                    lane_to_tmp.append(lane_from)
            adj_inter2from_lane[adj_id] = list(set(lane_to_tmp))
        return adj_inter2from_lane, adj_inter2target_road
    
    def _get_roadlinks(self):
        roadlinks = {}
        for lanelink in self.I.lanelinks:
            if not lanelink[0][2]:
                continue
            lane_from, lane_to, _ = lanelink[0]
            for road, lanes in self.I.road_lane_mapping.items():
                if lane_from in lanes:
                    road_from = road
                    lane_val = lane_from
                if lane_to in lanes:
                    road_to = road
            roadlinks[(road_from, road_to)] = lane_val
        return roadlinks
            
    def generate(self):
        fns = {fn: self.world.get_info(fn) for fn in self.fns}
        nums = {lane: 0 for lane in self.lanes}
        if self.num_intersections > 1:
            for adj_id, lane_froms in self.adj_inter2from_lane.items():
                for lane_from in lane_froms:
                    lane_vechile_info = fns['lane_vechile_info'][lane_from]
                    if lane_vechile_info == []:
                        continue
                    for vechile_info in lane_vechile_info:
                        name = vechile_info['name']
                        route = self.world.eng.vehicle.getRoute(name)
                        route_idx = self.world.eng.vehicle.getRouteIndex(name)
                        if route[route_idx + 1] == self.adj_inter2target_road[adj_id] and route_idx + 2 < len(route):
                            lane_target = self.roadlinks[(route[route_idx + 1], route[route_idx + 2])]
                            nums[lane_target] += 1
        return list(nums.values())