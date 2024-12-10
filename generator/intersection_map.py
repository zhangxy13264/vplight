import numpy as np
import math

class IntersectionMapGenerator():
    def __init__(self, world, I, fns=["pedestrian_position", "sidewalk_pedestrians"], targets=["walking_area_map"], area_width = 600, grid_width = 4):
        self.world = world
        self.I = I
        self.phase = I.current_phase

        self.world.subscribe(fns)
        self.fns = fns
        self.targets = targets
        
        self.result_functions = {
            "vehicle_map": self.vehicle_map,
            "pedestrian_map": self.pedestrian_map,
            "walking_area_map": self.walking_area_map,
            "walking_area_con": self.walking_area_con,
            "walking_area_num": self.walking_area_num,
        }
        self.area_width = area_width
        self.grid_width = grid_width

        self.vehicle_lanes = I.vehicle_lanes
        self.sidewalks = I.sidewalks
        
        self.ob_length = 0
        if ["walking_area_con"] == targets:
            self.ob_length = len(self.sidewalks) * 2
        elif ["walking_area_num"] == targets:
            self.ob_length = len(self.sidewalks)
        elif ["walking_area_map"] == targets:
            self.ob_length = len(self.sidewalks)
        
    def verify_id(self, trafficlight_id):
        def remove_prefix_if_exists(string, prefix):
            if string.startswith(prefix):
                return string[len(prefix):]
            return string
        junction_id = remove_prefix_if_exists(trafficlight_id,'GS_')
        return junction_id
    
    def _location_mapper(self, coordinate, area_length, area_width, grid_width):
        transformX = math.floor((coordinate[0] + area_length / 2) / grid_width)
        length_width_map = float(area_length) / area_width
        transformY = math.floor((coordinate[1] + area_width / 2) * length_width_map / grid_width)
        length_num_grids = int(area_length / grid_width)
        transformY = length_num_grids - 1 if transformY >= length_num_grids else transformY
        transformX = length_num_grids - 1 if transformX >= length_num_grids else transformX
        tempTransformTuple = (transformY, transformX)
        return tempTransformTuple
        
    def vehicle_map(self, fns):
        area_length = self.area_width
        area_width = self.area_width
        grid_width = self.grid_width
        
        junction_id = self.verify_id(self.I.id)
        junction_pos = self.world.eng.junction.getPosition(junction_id)
            
        length_num_grids = int(area_length / grid_width)
        mapOfCars = np.zeros((length_num_grids, length_num_grids))
        
        vehicle_position = fns["vehicle_position"] # 车辆到交叉口的距离
        lane_vehicles = fns["lane_vehicles"] # 车道上所有的车辆信息
        for lane in self.vehicle_lanes:
            
            for vehicle_info in lane_vehicles[lane]['vehicles']:
                if vehicle_info is None:
                    continue
                vehicle = vehicle_info['name']
                vehicle_pos = vehicle_position[vehicle]
                pos = (vehicle_pos[0] - junction_pos[0], vehicle_pos[1] - junction_pos[1])
                transform_tuple = self._location_mapper(pos, area_length, area_width, grid_width)  # transform the coordinates to location in grid
                mapOfCars[transform_tuple[0], transform_tuple[1]] = 1
        return mapOfCars       
        
    def pedestrian_map(self, fns):
        area_length = self.area_width
        area_width = self.area_width
        grid_width = self.grid_width
        
        junction_id = self.verify_id(self.I.id)
        junction_pos = self.world.eng.junction.getPosition(junction_id)
            
        length_num_grids = int(area_length / grid_width)
        mapOfPedestrian = np.zeros((length_num_grids, length_num_grids))
        
        pedestrian_position = fns["pedestrian_position"]
        lane_pedestrians = fns['sidewalk_pedestrians']
        for lane in self.sidewalks:
            for pedestrian_info in lane_pedestrians[lane]['pedestrians']:
                if pedestrian_info is None:
                    continue
                pedestrian = pedestrian_info['name']
                pedestrian_pos = pedestrian_position[pedestrian]
                pos = (pedestrian_pos[0] - junction_pos[0], pedestrian_pos[1] - junction_pos[1])
                transform_tuple = self._location_mapper(pos, area_length, area_width, grid_width)
                if mapOfPedestrian[transform_tuple[0], transform_tuple[1]] == 0:
                    mapOfPedestrian[transform_tuple[0], transform_tuple[1]] = 1
                else:
                    mapOfPedestrian[transform_tuple[0], transform_tuple[1]] += 1
        return mapOfPedestrian 
    
    def _average_coordinates(self, coordinates):
        x = 0
        y = 0
        for coordinate in coordinates:
            x += coordinate[0]
            y += coordinate[1]
        average = (x / len(coordinates), y / len(coordinates))
        return average
    
    def walking_area_map(self, fns):
        area_length = self.area_width
        area_width = self.area_width
        grid_width = self.grid_width

        sidewalks_maps = []
        sidewalks_cons = []
        for sidewalk_id in self.sidewalks:
            sidewalk_shape = self.world.eng.lane.getShape(sidewalk_id)
            sidewalk_avg_pos = self._average_coordinates(sidewalk_shape)

            length_num_grids = int(area_length / grid_width)
            mapOfPedestrians = np.zeros((length_num_grids, length_num_grids))
            
            pedestrian_position = fns["pedestrian_position"]
            lane_pedestrians = fns['sidewalk_pedestrians']
            for pedestrian_info in lane_pedestrians[sidewalk_id]['pedestrians']:
                if pedestrian_info is None:
                    continue
                pedestrian = pedestrian_info['name']
                pedestrian_pos = pedestrian_position[pedestrian]
                pos = (pedestrian_pos[0] - sidewalk_avg_pos[0], pedestrian_pos[1] - sidewalk_avg_pos[1])
                transform_tuple = self._location_mapper(pos, area_length, area_width, grid_width) 
                mapOfPedestrians[transform_tuple[0], transform_tuple[1]] = 1

            sidewalks_maps.append(mapOfPedestrians)
        return sidewalks_maps
    
    def walking_area_con(self, fns):
        sidewalks_cons = []
        for sidewalk_id in self.sidewalks:
            pedestrian_position = fns["pedestrian_position"]
            lane_pedestrians = fns['sidewalk_pedestrians']
            crossings = self.I.sidewalk2crossing[sidewalk_id]
            cross_con = {k:0 for k in crossings}
            for pedestrian_info in lane_pedestrians[sidewalk_id]['pedestrians']:
                if pedestrian_info is None:
                    continue
                pedestrian = pedestrian_info['name']
                next_edge = self.world.eng.person.getNextEdge(pedestrian)
                if next_edge in crossings:
                    cross_con[next_edge] += 1
            sidewalks_cons.append(cross_con)

        walking_area_con = [item for sublist in sidewalks_cons for item in sublist.values()]
        # walking_area_con = np.array(walking_area_con)
        return walking_area_con
    
    def walking_area_num(self, fns):
        sidewalks_nums = [0 for _ in range(len(self.sidewalks))]
        for sidewalk_id in self.sidewalks:
            lane_pedestrians = fns['sidewalk_pedestrians']
            for pedestrian_info in lane_pedestrians[sidewalk_id]['pedestrians']:
                if pedestrian_info is None:
                    continue
                sidewalks_nums[self.sidewalks.index(sidewalk_id)]+=1
        # sidewalks_nums = np.array(sidewalks_nums)
        return sidewalks_nums
            
    def generate(self):
        fns = {fn:self.world.get_info(fn) for fn in self.fns}
        ret = [self.result_functions[res](fns) for res in self.targets]
        if len(self.targets) == 1:
            ret = ret[0]
        return ret
