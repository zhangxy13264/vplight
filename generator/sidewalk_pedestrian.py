
import numpy as np
from world import world_sumo

class SidewalkPedestrianGenerator():
    def __init__(self, world, I, fns, in_only=False, average=None, negative=False):
        self.world = world
        self.I = I

        self.lanes = self.I.sidewalks
        self.world.subscribe(fns)
        self.fns = fns

        size = len(self.lanes)
        if average == "road":
            size = len(self.lanes)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size

        self.average = average
        self.negative = negative

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]
        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]

            if self.I.id in result:
                ret = np.append(ret, result[self.I.id])
                continue
            fn_result = np.array([])

            for sidewalk in self.lanes:
                fn_result = np.append(fn_result, np.array([result[sidewalk]]))
            ret = np.append(ret, fn_result)

        if self.negative:
            ret = ret * (-1)
        if self.average == "all":
            ret = np.array([np.mean(ret)])
        return ret

