from gaussian import get_next_point
import numpy as np
if __name__ == "__main__":
    data = [[1,3],[2,5],[10,6],[8,4],[3.5,7]]
    score = [5,8,11,4,2]
    act,gp_theta = get_next_point(np.array(data),np.array(score).T)
    print(act,gp_theta)