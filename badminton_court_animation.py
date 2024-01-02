import sys # 提供对Python解释器的访问和控制。
import argparse # 解析命令行参数和选项。
import heapq # 提供堆队列算法的实现。
import math # 提供常用的数学函数。
import random # 生成伪随机数。
import time # 提供时间相关的函数。
import numpy as np # 提供用于科学计算的多维数组对象和函数。
import matplotlib.pyplot as plt # 提供绘图函数。
import matplotlib.animation as animation # 提供绘制动画的函数。
import copy # 提供对象复制的函数。

# 常量定义
BDT_COURT_SCALE = 25 # 机器人扫球器的宽度是25cm，确保一次扫过全部捡起
# 以下为单个羽毛球场相关尺寸，按国际标准要求尺寸
BDT_COURT_LINE_WIDTH = 4 / BDT_COURT_SCALE
BDT_COURT_LENGTH_MAIN = 1340 / BDT_COURT_SCALE
BDT_COURT_LENGTH_ALL = 1730 / BDT_COURT_SCALE
BDT_COURT_LENGTH_BACK = 72 / BDT_COURT_SCALE 
BDT_COURT_LENGTH_MIDDLE = 388 / BDT_COURT_SCALE
BDT_COURT_LENGTH_FRONT = 198 / BDT_COURT_SCALE
BDT_COURT_WIDTH_MAIN = 610 / BDT_COURT_SCALE
BDT_COURT_WIDTH_ALL = 1010 / BDT_COURT_SCALE
BDT_COURT_WIDTH_SIDE = 42 / BDT_COURT_SCALE
BDT_COURT_WIDTH_MIDDLE = 253 / BDT_COURT_SCALE
BDT_COURT_ROW = 3 # 默认行场地个数，可以通过配置行进行变更
BDT_COURT_COLUMN = 3 # 默认列场地个数，可以通过配置行进行变更
ROBOT_VISIBLE_RANGE = int(300/BDT_COURT_SCALE-1) # 机器人可视范围为前后左右3米周围6米*6米
TOP_BADMINTON_COUNT = 10 # 路径最多有球点采样最多次数
COMPARE_RUN_COUNT = 10 # 算法性能比较时，默认比较次数，可以通过配置行进行变更

g_compare_flag = 0 # compare模式标志
g_demo_one_flag = 0 # demo_one模式标志
g_by_strategy = 'efficiency' # 策略为distance/efficiency优先
fig, ax = plt.subplots() # 画图全局对象

"""
Dijkstra 算法是一种广度优先搜索算法。它使用一个优先队列来存储待访问的节点，并根据已知的从起点到当前节点的距离对这些节点进行排>序。
在每一步中，它选择距离起点最近的节点，然后扩展该节点的所有邻居。
如果找到了到达某个邻居的更短路径，它就会更新相应的距离并将该邻居添加到优先队列中。
"""
def dijkstra(bg_grid, ball_grid, start, end, init_ball_count=0, init_path_len=0):
    width, height = len(bg_grid[0]), len(bg_grid)
    cost_map = [[math.inf for _ in range(width)] for _ in range(height)]
    cost_map[start[1]][start[0]] = 0
    priority_queue = [(0, start)]
    came_from = dict()

    while priority_queue:
        # pop最小cost节点
        current_cost, current_node = heapq.heappop(priority_queue)

        # 到达最终目标时停止搜索
        if current_node == end:
            break

        # 把当前节点的邻居节点以及cost加入优先队列
        for neighbor in get_neighbors(bg_grid, current_node):
            new_cost = current_cost + 1
            if new_cost < cost_map[neighbor[1]][neighbor[0]]:
                cost_map[neighbor[1]][neighbor[0]] = new_cost
                heapq.heappush(priority_queue, (new_cost, neighbor))
                came_from[neighbor] = current_node

    # came_from从叶子节点到根节点回溯，倒置、设置path信息
    path = reconstruct_path(came_from, start, end)
    path_all = set_path_info(ball_grid, path, start, end, init_ball_count, init_path_len)
    return path_all, cost_map[end[1]][end[0]]

# A*和Dijkstra算法的主要区别在于如何选择要扩展的下一个节点。Dijkstra 算法只关注从起点到当前节点的已知距离，而 A* 算法还关注从当前节点到目标节点的启发式距离。启发式函数使 A* 算法能够更有针对性地搜索最短路径，从而通常比 Dijkstra 算法更快。
# 从代码角度来看，两段代码的主要区别在于优先队列中的排序方式。在 Dijkstra 算法中，优先队列中的节点按照 current_cost 排序，而在 A* 算法中，它们按照 current_cost + heuristic(neighbor, end) 排序。这就是 A* 算法在选择要扩展的节点时考虑启发式信息的地方。
"""
A* 算法是一种启发式搜索算法，它在 Dijkstra 算法的基础上增加了一个启发式函数。
启发式函数用于估算从当前节点到目标节点的距离，以便更有针对性地搜索最短路径。
在每一步中，A* 算法选择具有最小总成本（已知距离 + 启发式距离）的节点进行扩展。
由于启发式函数的引入，A* 算法通常比 Dijkstra 算法更快地找到最短路径。
"""
def a_star(bg_grid, ball_grid, start, end, init_ball_count=0, init_path_len=0):
    width, height = len(bg_grid[0]), len(bg_grid)
    cost_map = [[math.inf for _ in range(width)] for _ in range(height)]
    cost_map[start[1]][start[0]] = 0
    priority_queue = [(0 + euclidean_distance(start, end), start)]
    came_from = dict()

    while priority_queue:
        # pop最小cost节点
        current_cost, current_node = heapq.heappop(priority_queue)

        # 到达最终目标时停止搜索
        if current_node == end:
            break

        # 把当前节点的邻居节点以及cost加入优先队列
        for neighbor in get_neighbors(bg_grid, current_node):
            new_cost = cost_map[current_node[1]][current_node[0]] + 1
            if new_cost < cost_map[neighbor[1]][neighbor[0]]:
                cost_map[neighbor[1]][neighbor[0]] = new_cost
                heapq.heappush(priority_queue, (new_cost + euclidean_distance(neighbor, end), neighbor)) # 在 Dijkstra 算法的基础上增加了一个启发式函数
                came_from[neighbor] = current_node

    # came_from从叶子节点到根节点回溯，倒置、设置path信息
    path = reconstruct_path(came_from, start, end)
    path_all = set_path_info(ball_grid, path, start, end, init_ball_count, init_path_len)
    return path_all, cost_map[end[1]][end[0]]

# 捡球效率优先算法，从起点到目标终点
def pathfinding(balls_grid, obstacles_grid, start, goal, init_ball_count=0, init_path_len=0):
    # Initialize the open and closed sets
    open_set = [(0, start)]  # (f, (x, y)) # open_set用于存储已经被发现但尚未被探索的节点，目的是根据节点的f值（路径成本和启发式价值的总和）确定下一个要探索的节点
    closed_set = set() # closed_set用于存储已经被探索的节点的集合，目的是防止算法探索已经被探索过的节点，这可能会导致低效和无限循环
    # Initialize the dictionary to keep track of the path
    came_from = {} # 记录自己的前一个节点，也就是父亲节点，我是哪里来
    # Initialize the dictionary to keep track of the cost of the path
    cost_so_far = {} # 记录从起点到达当前节点的cost（这里记录球数和路程数）
    cost_so_far[start] = (balls_grid[start[1]][start[0]], 0) # 所经过路径：到当前节点为止的球数和路径长度

    while open_set:
        current = heapq.heappop(open_set)[1] # Pop the node with the lowest f value
        closed_set.add(current) # Add the current node to the closed set
        if current == goal: # Check if the current node is the goal
            # Reconstruct the path, set path info
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            path_all = set_path_info(balls_grid, path, start, goal, init_ball_count, init_path_len)
            return path_all

        # Get the neighbors of the current node
        neighbors = []
        for neighbor in get_neighbors(obstacles_grid, current):
            if neighbor not in closed_set:
                neighbors.append(neighbor)

        # add the via node to the heuristic, 也就是会导向除目标点外，也会考虑更近更有价值的途径点
        via = goal
        if g_compare_flag == 0 and g_by_strategy == 'distance':
            # 仅仅考虑路程远近，找最近的分块作为途径点
            min_heuristic = euclidean_distance2(current[0], current[1], via[0], via[1])
            for viewBall in get_viewBalls(balls_grid, obstacles_grid, current, goal):
                if viewBall not in closed_set and viewBall not in came_from:
                    new_heuristic = euclidean_distance2(current[0], current[1], viewBall[0], viewBall[1])
                    if (new_heuristic < min_heuristic):
                        min_heuristic = new_heuristic
                        via = viewBall
        elif g_compare_flag == 0 and g_by_strategy == 'efficiency':
            # 考虑路径远近与球数结合的效率，找捡球效率最高的分块作为途径点
            via_distance = euclidean_distance2(current[0], current[1], via[0], via[1])
            via_ball_count = balls_grid[via[1]][via[0]] # 不考虑中间是否有球
            max_efficiency_reverse = via_distance/via_ball_count if via_ball_count != 0 else float('inf') 
            for viewBall in get_viewBalls(balls_grid, obstacles_grid, current, goal):
                if viewBall not in closed_set and viewBall not in came_from:
                    new_via_distance = euclidean_distance2(current[0], current[1], viewBall[0], viewBall[1])
                    new_via_ball_count = balls_grid[viewBall[1]][viewBall[0]]
                    new_max_efficiency_reverse = new_via_distance/new_via_ball_count if new_via_ball_count != 0 else float('inf')
                    if (new_max_efficiency_reverse < max_efficiency_reverse):
                        max_efficiency_reverse = new_max_efficiency_reverse
                        via = viewBall # 不断循环，找到捡球效率最高的点作为途径点

        # Loop through the neighbors
        for neighbor in neighbors:
            # 尝试算下Calculate the cost of the path to the neighbor
            new_path_len = cost_so_far[current][1] + euclidean_distance2(current[0], current[1], neighbor[0], neighbor[1])
            new_ball_count = cost_so_far[current][0] + balls_grid[neighbor[1]][neighbor[0]]
            new_cost = new_path_len / new_ball_count if new_ball_count != 0 else float('inf')
            if neighbor not in cost_so_far: # 所有当前节点的邻居均会被探索，放入open_set中，更新信息
                cost_so_far[neighbor] = (new_ball_count, new_path_len) #new_cost
                # Calculate the f/heuristic value of the neighbor
                heuristic1 = euclidean_distance2(neighbor[0], neighbor[1], via[0], via[1])
                #heuristic2 = euclidean_distance2(via[0], via[1], goal[0], goal[1]) # 暂时不考虑和终点之间的距离，因为目标不是尽快到目标点，而是附近途径点有球也会导致效率更高
                f = heuristic1 / balls_grid[neighbor[1]][neighbor[0]] if balls_grid[neighbor[1]][neighbor[0]] !=0 else heuristic1*2 # 不能是inf，因为周边没球就都是inf，么有区分度，所有用0.5
                # Add the neighbor to the open set
                heapq.heappush(open_set, (f, neighbor)) #heapq.heappush(open_set, (heuristic1, neighbor))
                # Update the dictionary to keep track of the path
                came_from[neighbor] = current # 记录我的父亲节点
    return {} # No path found

# 根据算法和策略，把全场捡完
def demo_all(bg_grid, ball_grid, start, end, demo_algorithm='pathfinding'):
    path_all = [] # 记录完整路径
    vias = [] # 记录有球途径点
    vias.append(start)
    current = start
    init_ball_count = 0
    init_path_len =  1
    next_point = get_next_point(bg_grid, ball_grid, start)
    while next_point != current:
        vias.append(next_point)
        path_seg = path_algorithm(ball_grid, bg_grid, current, next_point, init_ball_count, init_path_len, demo_algorithm)
        for node_temp in path_seg: # 走过则把球捡起来，置为零
            ball_grid[node_temp[0][1]][node_temp[0][0]] = 0
        path_all.extend(path_seg)
        init_ball_count = path_all[-1][2] # path_all到目前为止的最后一个球数计数（因为球数都是累加的）
        init_path_len = path_all[-1][3] + 1 # path_all到目前为止的最后一个路径值计数（因为路径值都是累加的）
        current = next_point # 进入下一段
        next_point = get_next_point(bg_grid, ball_grid, current)

    print("demo_all", demo_algorithm, "by", g_by_strategy, "vias:", len(vias), "steps:", len(path_all))
    return path_all

# 根据输入进行算法选择
def path_algorithm(ball_grid, bg_grid, current, next_point, init_ball_count, init_path_len, demo_algorithm='pathfinding'):
    if demo_algorithm == 'dijkstra':
        path_seg, cost_map = dijkstra(bg_grid, ball_grid, current, next_point, init_ball_count, init_path_len)
        return path_seg
    elif demo_algorithm == 'astar':
        path_seg, cost_map = a_star(bg_grid, ball_grid, current, next_point, init_ball_count, init_path_len)
        return path_seg
    else:
        return pathfinding(ball_grid, bg_grid, current, next_point, init_ball_count, init_path_len)

# 根据不同策略，选择下一个目标点
def get_next_point(bg_grid, ball_grid, start):
    if g_by_strategy == 'distance':
        return get_next_shortest_distance_point(bg_grid, ball_grid, start)
    else:
        return get_next_best_efficient_point(bg_grid, ball_grid, start)

# 根据起点，按欧式距离，效率优先，获得下一个最优目标点
def get_next_best_efficient_point(bg_grid, ball_grid, start):
    vias = []
    best_efficient = 0
    best_point = start
    for x in range(len(ball_grid)):
        for y in range(len(ball_grid[0])):
            if ball_grid[x][y] > 0 and bg_grid[x][y] == 0:
                current = (y, x)
                new_pathj_len = euclidean_distance(start, current)
                new_ball_count = ball_grid[x][y]
                new_efficient = new_ball_count/new_pathj_len if new_pathj_len != 0 else 0
                if new_efficient > best_efficient: # 找效率最大的
                    best_point = current
                    best_efficient = new_efficient
    return best_point

# 根据起点，按欧式距离，距离优先，获得下一个最近目标点
def get_next_shortest_distance_point(bg_grid, ball_grid, start):
    vias = []
    shortest_distance = float('inf')
    best_point = start
    for x in range(len(ball_grid)):
        for y in range(len(ball_grid[0])):
            if ball_grid[x][y] > 0 and bg_grid[x][y] == 0:
                current = (y, x)
                new_path_len = euclidean_distance(start, current)
                if new_path_len < shortest_distance: # 着路径最短的
                    best_point = current
                    shortest_distance = new_path_len 
    return best_point

# 设置路径每一个点的颜色、球数、路径长度、结束标志
def set_path_info(ball_grid, path, start, end, init_ball_count=0, init_path_len=0):
    path_all = []
    ball_count = 0
    path_len = 0
    gray_color = [192, 192, 192] # 试算路径,用灰色
    previous_node = start
    for path_i in path:
        ball_count += ball_grid[path_i[1]][path_i[0]]
        path_len += euclidean_distance2(path_i[0], path_i[1], previous_node[0], previous_node[1])
        endflag = 0
        if path_i == end: # 每一段结束那点，作一个end标志
            endflag = 1
        node = (path_i, gray_color, ball_count+init_ball_count, path_len+init_path_len, endflag)
        path_all.append(node)
        previous_node = path_i
    return path_all

# 重画效率最优路径, 也就是把效率最高的那一段加在最后，且用不同颜色
def redraw_best_path_info(path_all):
    if g_compare_flag == 1: # compare模式的时候，和展示无关
        return None

    # 找到效率最高的哪个end点
    red_color = [255, 0, 0] # 效率最高路径
    best_efficiency = 0
    best_path_end = []
    best_path = []
    for path_i in path_all:
        if path_i[4] == 1:
            new_efficiency = path_i[2]/path_i[3] if path_i[3] != 0 else 0
            if new_efficiency > best_efficiency:
                best_efficiency = new_efficiency
                best_path_end = path_i

    # 到path_all中把效率最高的那段没个点都找出来，从end位置开始，所以是倒序
    reset_flag = 0
    for path_j in path_all[::-1]:
        if path_j[0] == best_path_end[0] and path_j[4] == 1: # 是end位置，且end等于效率最高的那个end，表示找到了
            reset_flag = 1
            node = (path_j[0], red_color, path_j[2], path_j[3], path_j[4])
            best_path.append(node)
            continue
        if reset_flag == 0: # 还没找到，则跳过
            continue
        if path_j[4] == 0: # 在找到最优那段后，全部计入，一直到下一个有end标志的点
            node = (path_j[0], red_color, path_j[2], path_j[3], path_j[4])
            best_path.append(node)
        else:
            break   

    best_path.reverse() # 反过来找的，所以需要反过来
    path_all.extend(best_path)
    return best_path_end[0]

# 获取可视区域内有球坐标点
def get_viewBalls(balls_grid, obstacles_grid, current, goal):
    viewBalls = []
    for i in range(ROBOT_VISIBLE_RANGE, (0-ROBOT_VISIBLE_RANGE-1), -1):
        for j in range(ROBOT_VISIBLE_RANGE, (0-ROBOT_VISIBLE_RANGE-1), -1):
            if (goal[0] > current[0] and i < 0) or (goal[0] < current[0] and i > 0) \
                or (goal[1] > current[1] and j < 0) or (goal[1] < current[1] and j > 0): # 不同方向
                if abs(i) > ROBOT_VISIBLE_RANGE/2 or abs(j) > ROBOT_VISIBLE_RANGE/2: # 且范围过半
                    continue # 只捡附近的或者同方向的球
            x = current[0] + i
            y = current[1] + j
            if (0 <= y < len(balls_grid) and 0 <= x < len(balls_grid[0]) and obstacles_grid[y][x] == 0 and
                (balls_grid[y][x] > 0 )):
                viewBalls.append((x, y))
    return viewBalls

def get_neighbors(grid, node):
    x, y = node
    width, height = len(grid[0]), len(grid)
    neighbors = [(x+dx, y+dy) for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]]
    # 去掉障碍物和边界
    valid_neighbors = [(x, y) for x, y in neighbors if 0 <=
                       x < width and 0 <= y < height and grid[y][x] == 0]
    return valid_neighbors

def get_neighbors2(bg_grid, row, col):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        r, c = row + dr, col + dc
        if 0 <= r < len(bg_grid) and 0 <= c < len(bg_grid[0]):
            yield r, c

# 从叶子阶段找到跟节点，倒置
def reconstruct_path(came_from, start, end):
    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    return path[::-1]

def heuristic(ball_grid, p1, p2):
    return distance_and_ball_number(ball_grid, p1, p2)

def heuristic2(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# p1 is current node, p2 is end node
def distance_and_ball_number(ball_grid, p1, p2):
    distance = euclidean_distance(p1, p2)
    p1x, p1y = p1
    n = ball_grid[p1y][p1x]
    return math.ceil(distance/(n+1))

def manhattan_distance(p1, p2):
    """
    启发式（heuristic）函数，曼哈顿距离。
    这种启发式函数适用于只能在水平和垂直方向上移动的场景。
    """
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def manhattan_distance2(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(p1, p2):
    """
    启发式（heuristic）函数，欧几里得距离。
    这种启发式函数适用于可以沿任意方向移动的场景。
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def euclidean_distance2(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    #return math.sqrt(((x1 - x2)**2 + (y1 - y2)**2)/2)

# 概率生成随机障碍与固定障碍，这里仅仅用在compare中
def generate_bg_grid(width, height, obstacle_ratio):
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for n1 in range(1, 5):
        for n2 in range(4, 8):
            grid[height-n1][width-n2] = 1
    for y in range(height):
        for x in range(width):
            # 增加两侧固定障碍
            obstacle_h_s1 = (BDT_COURT_WIDTH_ALL-BDT_COURT_WIDTH_MAIN)/2
            obstacle_h_e1 = obstacle_h_s1 + 0.05 * height
            obstacle_h_e2 = obstacle_h_s1 + BDT_COURT_WIDTH_MAIN
            obstacle_h_s2 = obstacle_h_e2 - 0.05 * height
            middle_w = 0.5 * width
            obstacle_w = 0.05 * width
            if (obstacle_h_s1 < y < obstacle_h_e1) or (obstacle_h_s2 < y < obstacle_h_e2):
                if (x > (middle_w - obstacle_w)) and (x < (middle_w + obstacle_w)):
                    grid[y][x] = 1
            # 增加随机障碍
            if random.random() < obstacle_ratio:
                grid[y][x] = 1
                # 查找附近节点，同时将其设置为障碍物，增大障碍物体积
                for neighbor in get_neighbors(grid, (x, y)):
                    grid[neighbor[1]][neighbor[0]] = 1 # random.randint(0, 1) # 随机一下，生成一些不规则形状障碍
    return grid

# 生成每个球场的固定障碍物，目前只有球柱
def generate_grid_obstacle_fixed(grid, origin_x, origin_y, width, height):
    #grid = [[0 for _ in range(width)] for _ in range(height)]
    #for n1 in range(1, 5):
    #    for n2 in range(4, 8):
    #        grid[origin_y+height-n1][origin_x+width-n2] = 1
    for y in range(height):
        for x in range(width):
            # 增加中间网两侧固定障碍
            obstacle_h_s1 = (BDT_COURT_WIDTH_ALL-BDT_COURT_WIDTH_MAIN)/2
            obstacle_h_e1 = obstacle_h_s1 + 0.05 * height
            obstacle_h_e2 = obstacle_h_s1 + BDT_COURT_WIDTH_MAIN
            obstacle_h_s2 = obstacle_h_e2 - 0.05 * height
            middle_w = 0.5 * width
            obstacle_w = 0.05 * width
            if (obstacle_h_s1 < y < obstacle_h_e1) or (obstacle_h_s2 < y < obstacle_h_e2):
                if (x > (middle_w - obstacle_w)) and (x < (middle_w + obstacle_w)):
                    grid[origin_y+y][origin_x+x] = 1

# 概率生成整个大片球场的障碍物
def generate_grid_obstacle_ratio(grid, width, height, obstacle_ratio):
    #grid = [[0 for _ in range(width)] for _ in range(height)]
    #for n1 in range(1, 5):
    #    for n2 in range(4, 8):
    #        grid[height-n1][width-n2] = 1
    for y in range(height):
        for x in range(width):
            # 增加随机障碍
            if random.random() < obstacle_ratio:
                grid[y][x] = 1
                # 查找附近节点，同时将其设置为障碍物，增大障碍物体积
                for neighbor in get_neighbors(grid, (x, y)):
                    grid[neighbor[1]][neighbor[0]] = 1 # random.randint(0, 1) # 随机一下，生成一些不规则形状障碍

# 概率生成整个大片球场的羽毛球
def generate_ball_number_grid(width, height, ball_ratio):
    grid = [[0 for _ in range(width)] for _ in range(height)]
    #grid = np.zeros([height,width])
    # for n1 in range(1, 5):
    #     for n2 in range(4, 8):
    #         grid[height-n1][width-n2] = 1
    total_balls = 0
    for y in range(height):
        for x in range(width):
            if random.random() < ball_ratio:
                grid[y][x] = random.randint(0, 9)
                total_balls += grid[y][x]
    print("total_balls:", total_balls)
    return grid

# 获得ball_grid中top10球多的坐标点
def get_top_end_loc(ball_grid, bg_grid, top_num=TOP_BADMINTON_COUNT):
    ball_grid1 = np.copy(ball_grid)
    bg_grid1 = np.array(bg_grid)
    ball_grid1[bg_grid1 == 1] = -1 # 将bg_grid中值为1的坐标点在ball_grid中的值设为-1
    sorted_indices = np.argsort(ball_grid1.flatten()) # 将二维数组打平并排序，返回排序后的索引值
    top_count = top_num if sorted_indices.size >= top_num else sorted_indices.size #int(np.ceil(sorted_indices.size * 0.001)) # 计算要选取的前10的数量，并将其向上取整
    top_indices = np.unravel_index(sorted_indices[-top_count:], ball_grid1.shape) # 选取排序后的索引值中最大的前10%的值所对应的下标
    return top_indices # 输出选取的值和它们的下标

# 画单个羽毛球场，(origin_x, origin_y)表示左上角其实坐标
def create_badminton_court_background(origin_x, origin_y):
    length = int(BDT_COURT_LENGTH_ALL-1)
    width = int(BDT_COURT_WIDTH_ALL-1)
    linewidth = int(BDT_COURT_LINE_WIDTH)
    main_start_x = origin_x + int((BDT_COURT_LENGTH_ALL - BDT_COURT_LENGTH_MAIN)/2)
    main_start_y = origin_y + int((BDT_COURT_WIDTH_ALL - BDT_COURT_WIDTH_MAIN)/2)
    line_color = 'w-' # 白色实线
    line_color_other = 'k-' # 黑色实线

    # 创建一个空白的白色图像
    #background = np.full((width, length, 3), 255, dtype=np.uint8)

    # 画包括副场地的整个场地
    #ax.plot([x1, x2], [y1, y2], line_color_other, 1)
    ax.plot([origin_x, origin_x+length], [origin_y, origin_y], line_color_other, 1)
    ax.plot([origin_x, origin_x+length], [origin_y+width, origin_y+width], line_color_other, 1)
    ax.plot([origin_x, origin_x], [origin_y, origin_y+width], line_color_other, 1)
    ax.plot([origin_x+length, origin_x+length], [origin_y, origin_y+width], line_color_other, 1)

    # 画羽毛球场地的边线
    ax.plot([main_start_x, main_start_x], [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)
    ax.plot([main_start_x+BDT_COURT_LENGTH_MAIN, main_start_x+BDT_COURT_LENGTH_MAIN], [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)
    ax.plot([main_start_x, main_start_x+BDT_COURT_LENGTH_MAIN], [main_start_y, main_start_y], line_color, linewidth)
    ax.plot([main_start_x, main_start_x+BDT_COURT_LENGTH_MAIN], [main_start_y+BDT_COURT_WIDTH_MAIN, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)

    # 画水平方向的中间虚线
    ax.plot([origin_x + (length // 2), origin_x + (length // 2)],
            [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], 'w--', linewidth)

    # 画服务线(单打)
    singles_service_line = BDT_COURT_WIDTH_SIDE
    ax.plot([main_start_x, main_start_x+BDT_COURT_LENGTH_MAIN], [main_start_y + singles_service_line,
            main_start_y + singles_service_line], line_color, linewidth)
    ax.plot([main_start_x, main_start_x+BDT_COURT_LENGTH_MAIN], [main_start_y + BDT_COURT_WIDTH_MAIN - singles_service_line,
            main_start_y + BDT_COURT_WIDTH_MAIN - singles_service_line], line_color, linewidth)

    # 左右两边区域的水平中线
    middle_height = origin_y + 0.5 * width
    middle_x_1 = main_start_x + BDT_COURT_LENGTH_BACK + BDT_COURT_LENGTH_MIDDLE + BDT_COURT_LINE_WIDTH*2
    middle_x_2 =  middle_x_1 + BDT_COURT_LENGTH_FRONT*2
    middle_x_3 = main_start_x + BDT_COURT_LENGTH_MAIN
    ax.plot([main_start_x, middle_x_1], [middle_height,
                              middle_height], line_color, linewidth)
    ax.plot([middle_x_2, middle_x_3], [middle_height,
                                  middle_height], line_color, linewidth)

    # 画后服务线
    singles_line = BDT_COURT_LENGTH_BACK
    ax.plot([main_start_x+singles_line, main_start_x+singles_line], [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)
    back_x = main_start_x + BDT_COURT_LENGTH_MAIN - BDT_COURT_LENGTH_BACK
    ax.plot([back_x, back_x],
            [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)

    # 画前服务线
    #singles_line = 0.32 * width
    front_x1 = main_start_x + BDT_COURT_LENGTH_BACK + BDT_COURT_LENGTH_MIDDLE + BDT_COURT_LINE_WIDTH*2
    front_x2 = middle_x_1 + BDT_COURT_LENGTH_FRONT*2
    ax.plot([front_x1, front_x1], [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)
    ax.plot([front_x2, front_x2], [main_start_y, main_start_y+BDT_COURT_WIDTH_MAIN], line_color, linewidth)

# 可视化路径演示
def visualize_path(bg_grid, ball_grid, path):
    # 在Matplotlib中的Axes对象ax上显示一个图像，图像的数据存储在变量bg_grid中。
    # 参数origin="upper"表示选择图像的原点为图像的左上角。
    # 参数animated=True表示可以对这个图像进行动画操作。
    # 参数alpha=0.618表示设置图像的透明度为0.618。
    im = ax.imshow(bg_grid, origin="upper", animated=True, alpha=0.618)

    # 转换成三通道颜色
    new_grid = [[[0 for _ in range(3)] for _ in range(len(bg_grid[0]))]
                for _ in range(len(bg_grid))]

    # 显示球场上分布的羽毛球球数
    styles = dict(ha="center", va="center", color="black", size=8)
    for x in range(len(bg_grid)):
        for y in range(len(bg_grid[0])):
            if ball_grid[x][y] > 0:
                ax.annotate(str(ball_grid[x][y]), xy=(y, x), **styles)
            new_grid[x][y] = [39*(1-bg_grid[x][y]), 174 * (1-bg_grid[x][y]), 96*(1-bg_grid[x][y])]

    # 动画展示路径
    styles2 = dict(ha="left", va="top", color="green", size=18) 
    text = ax.annotate('', xy=(0, BDT_COURT_WIDTH_ALL * BDT_COURT_ROW), xytext=(0, 0), textcoords='offset points', **styles2)
    def update(frame): # 这是一个回调函数，也就是图像展示的时候每过一帧会调用一次，frame从0开始
        path_len = round(path[frame][3], 2)
        efficiency = round(path[frame][2]/path[frame][3] if path[frame][3] != 0 else 0, 2)
        #print(str(frame)+"/"+str(len(path)), path[frame][0], str(path[frame][2])+"/"+str(path_len)+"="+str(efficiency))
        current = path[frame][0]
        #new_grid[current[1]][current[0]] = [255, 0, 0]
        new_grid[current[1]][current[0]] = path[frame][1]
        im.set_array(new_grid)
        # 一条路径跑完就显示路径球数，路径长度信息
        if path[frame][4] == 1:
            s=str(path[frame][0])+str(path[frame][2])+"/"+str(path_len)+"="+str(efficiency)
            if g_demo_one_flag != 0:
                styles3 = dict(ha="left", va="top", color="black", size=8)
                ax.annotate(s, xy=(current[0], current[1]), **styles3)
            text.set_text(s) # 实现覆盖效果
            print(s)
        if frame == len(path)-1: # 画红圈标识结束
            ax.plot(current[0], current[1], 'ro')
            ax.annotate("end", xy=(current[0], current[1]))
        return [im]

    #- fig 是一个 matplotlib.figure.Figure 对象，表示要绘制的图形。
    #- update 是一个回调函数，每次更新图形时都会被调用。
    #- frames 是一个整数，表示动画的帧数。
    #- interval 是一个整数，表示每一帧之间的间隔时间，单位是毫秒。
    #- blit 是一个布尔值，表示是否使用 blitting 优化绘制速度。
    #- repeat 是一个布尔值，表示动画是否应该重复播放。
    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=1, blit=True, repeat=False)
    plt.show()

# demo_all模式：把全场球捡完模式
# demo_one模式：尝试球最多的top10路径作为终点，选择其中捡球效率最高的那条
def badminton_court(demo_mode='demo_all', demo_algorithm='pathfinding'):
    width = int(BDT_COURT_LENGTH_ALL * BDT_COURT_COLUMN)
    height = int(BDT_COURT_WIDTH_ALL * BDT_COURT_ROW)
    obstacle_ratio = 0.001 # 随机生成障碍物概率
    ball_ratio = 0.007 # 随机生成球概率

    # 创建一个空白的白色图像, 3色RGB
    background = np.full((width, height, 3), 255, dtype=np.uint8)

    # 循环行列，每次画一个单场地
    for n2 in range(0, BDT_COURT_ROW):
        for n1 in range(0, BDT_COURT_COLUMN):
            start_x = int(n1*BDT_COURT_LENGTH_ALL)
            start_y = int(n2*BDT_COURT_WIDTH_ALL)
            create_badminton_court_background(start_x,start_y)

    # 显示并保存背景图
    ax.imshow(background, origin='upper')
    ax.axis('scaled')
    ax.axis('off')

    # 随机生成一个羽毛球网格
    ball_grid = generate_ball_number_grid(width, height, ball_ratio)
    visualize_ball_grid = copy.deepcopy(ball_grid) # 后面纯展示用，因为每次捡完球ball_grid将会被清空
    print("field size:", width, "*", height)
    
    # 随机生成一个障碍物网格，其中固定障碍物为球场球柱
    bg_grid = [[0 for _ in range(width)] for _ in range(height)]
    generate_grid_obstacle_ratio(bg_grid, width, height, obstacle_ratio)
    for n2 in range(0, BDT_COURT_ROW):
        for n1 in range(0, BDT_COURT_COLUMN):
            start_x = int(n1*BDT_COURT_LENGTH_ALL)
            start_y = int(n2*BDT_COURT_WIDTH_ALL)
            generate_grid_obstacle_fixed(bg_grid, start_x, start_y, int(BDT_COURT_LENGTH_ALL), int(BDT_COURT_WIDTH_ALL))

    # 设置起点和终点
    #start = (110, 60)
    start = (random.randint(1,width-1), random.randint(1,height-1)) 
    print("start:", start)

    if demo_mode == 'demo_one':
        # 取top10作为终点
        top_indices = get_top_end_loc(ball_grid, bg_grid)

        # 使用不同算法找到最佳路径
        path = []
        ends = []
        for i in range(len(top_indices[0])-1, -1, -1):
            end = (top_indices[1][i], top_indices[0][i])
            ends.append(end)
            if demo_algorithm == 'dijkstra':
                path_temp, cost = dijkstra(bg_grid, ball_grid, start, end)
            if demo_algorithm == 'astar':
                path_temp, cost = a_star(bg_grid, ball_grid, start, end)
            if demo_algorithm == 'pathfinding':
                path_temp = pathfinding(ball_grid, bg_grid, start, end)
            path.extend(path_temp)
        # 最后选择的一条，用不同颜色真正走一遍
        redraw_best_path_info(path)
    
    else: # demo_all
        end = start
        path = demo_all(bg_grid, ball_grid, start, end, demo_algorithm)

    # 可视化路径
    ax.plot(start[0], start[1], 'ro') # 再start画红圈
    ax.annotate("start", xy=(start[0], start[1])) # 展示"start"
    visualize_path(bg_grid, visualize_ball_grid, path) # 展示路径演示效果
    
# 不同算法效率对比,并用折线图展示出来
def compare_algorithms(compare_num=COMPARE_RUN_COUNT):
    width, height = 200, 100
    obstacle_ratio = 0.01
    ball_ratio = 0.02
    global g_by_strategy

    # 随机生成一个网格
    bg_grid = generate_bg_grid(width, height, obstacle_ratio)
    ball_grid = generate_ball_number_grid(width, height, ball_ratio)
    reuse_ball_grid = copy.deepcopy(ball_grid) # 后面轮次用，因为每一轮，球都会被捡完，也就是ball_grid被清空 

    # 固定起点
    start = (0, 0)

    # 取top作为终点 
    top_indices = get_top_end_loc(ball_grid, bg_grid, compare_num)
    ends = [] 
    for i in range(len(top_indices[0])-1, -1, -1):
        end = (top_indices[1][i], top_indices[0][i])
        ends.append(end) 
    print("start:", start, "ends:", len(ends), ends)

    dijkstra_times1 = []
    a_star_times1 = []
    pathfinding_times1 = []
    dijkstra_times2 = []
    a_star_times2 = []
    pathfinding_times2 = []
    count = 0
    #for end in end_positions:
    for end in ends:
        if bg_grid[end[1]][end[0]] == 1: # 跳过end在障碍物上的情况
            print("jump a end", end, "which is on the obstacle")
            continue

        count += 1
        print("Comparison Round" + str(count) + ":")

        # by distance
        g_by_strategy = 'distance'
        ball_grid = copy.deepcopy(reuse_ball_grid) # 因为每一轮，球都会被捡完，也就是ball_grid被清空,需要重新赋值，深度copy  
        start_time = time.time()
        demo_all(bg_grid, ball_grid, start, end, 'dijkstra')
        dijkstra_times1.append(time.time() - start_time) # 记录dijkstra跑完全场所花费时间
        ball_grid = copy.deepcopy(reuse_ball_grid) # 因为每一轮，球都会被捡完，也就是ball_grid被清空,需要重新赋值，深度copy
        start_time = time.time()
        demo_all(bg_grid, ball_grid, start, end, 'astar')
        a_star_times1.append(time.time() - start_time) # 记录a_star跑完全场所花费时间
        ball_grid = copy.deepcopy(reuse_ball_grid) # 因为每一轮，球都会被捡完，也就是ball_grid被清空,需要重新赋值，深度copy
        start_time = time.time()
        demo_all(bg_grid, ball_grid, start, end, 'pathfinding')
        pathfinding_times1.append(time.time() - start_time) # 记录pathfinding跑完全场所花费时间

        # by efficiency
        g_by_strategy = 'efficiency'
        ball_grid = copy.deepcopy(reuse_ball_grid) # 因为每一轮，球都会被捡完，也就是ball_grid被清空,需要重新赋值，深度copy
        start_time = time.time()
        demo_all(bg_grid, ball_grid, start, end, 'dijkstra')
        dijkstra_times2.append(time.time() - start_time) # 记录dijkstra跑完全场所花费时间
        ball_grid = copy.deepcopy(reuse_ball_grid) # 因为每一轮，球都会被捡完，也就是ball_grid被清空,需要重新赋值，深度copy
        start_time = time.time()
        demo_all(bg_grid, ball_grid, start, end, 'astar')
        a_star_times2.append(time.time() - start_time) # 记录a_star跑完全场所花费时间
        ball_grid = copy.deepcopy(reuse_ball_grid) # 因为每一轮，球都会被捡完，也就是ball_grid被清空,需要重新赋值，深度copy
        start_time = time.time()
        demo_all(bg_grid, ball_grid, start, end, 'pathfinding')
        pathfinding_times2.append(time.time() - start_time) # 记录pathfinding跑完全场所花费时间

    # 绘制对比折线图
    x = list(range(1, len(dijkstra_times1) + 1))
    plt.plot(x, dijkstra_times1, label='Dijkstra by distance')
    plt.plot(x, dijkstra_times2, label='Dijkstra by efficiency')
    plt.plot(x, a_star_times1, label='A* by distance')
    plt.plot(x, a_star_times2, label='A* by efficiency')
    plt.plot(x, pathfinding_times1, label='Pathfinding by distance')
    plt.plot(x, pathfinding_times2, label='Pathfinding by efficiency')
    plt.xlabel('Comparison Round')
    plt.ylabel('Execution Time (s)')
    plt.title('Dijkstra vs A* vs Pathfinding')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_model = input("run demo_all/demo_one/compare?")
    # 通过把球放置在不同落点，比较算法对相同起点终点的寻址时间(效率对比)
    if run_model == 'compare':
        g_compare_flag = 1
        compare_num = int(input("compare_num?(n)"))
        compare_algorithms(compare_num)

    else:
        algorithms = input("with dijkstra/astar/pathfinding?")
        BDT_COURT_ROW = int(input("ROW?(n)"))
        BDT_COURT_COLUMN = int(input("COLUMN?(n)"))
        g_by_strategy = input("by distance/efficiency?)")

        # 演示demo程序，捡全场的球
        if run_model == 'demo_all':
            badminton_court('demo_all', algorithms)

        # 演示算法，试算起点到终点top路径，选择最优一条
        elif run_model == 'demo_one':
            g_demo_one_flag = 1
            badminton_court('demo_one', algorithms)

        else:
            print("Input error!")

'''
# 主程序入口
if __name__ == '__main__':
    # 获取命令行参数, nargs='?'表示可选参数但有的话后面带值，nargs='*'表示可选参数且后面可以不带值
    #args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    #parser.add_argument('--demo_all', nargs='*', default=None, help='demo_all')
    parser.add_argument('--demo_all', nargs='?', type=str, default=None, help='demo_all dijkstra/astar/pathfinding')
    parser.add_argument('--demo_one', nargs='?', type=str, default=None, help='demo_one dijkstra/astar/pathfinding')
    parser.add_argument('--by', nargs='?', type=str, default=None, help='by distance/efficiency')
    parser.add_argument('--row', nargs='?', type=int, default=None, help='row n')
    parser.add_argument('--column', nargs='?', type=int, default=None, help='column n')
    parser.add_argument('--compare', nargs='?', type=int, default=None, help='compare n')
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()

    if args.row is not None:
        BDT_COURT_ROW = args.row

    if args.column is not None:
        BDT_COURT_COLUMN = args.column

    if args.by is not None:
        g_by_strategy = args.by

    # 演示demo程序，捡全场的球
    if args.demo_all is not None:
        badminton_court('demo_all', args.demo_all)

    # 演示算法，试算起点到终点top路径，选择最优一条
    if args.demo_one is not None:
        g_demo_one_flag = 1
        badminton_court('demo_one', args.demo_one)

    # 通过把球放置在不同落点，比较算法对相同起点终点的寻址时间(效率对比)
    if args.compare is not None:
        g_compare_flag = 1
        compare_num = args.compare
        compare_algorithms(compare_num)
'''
