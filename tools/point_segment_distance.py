'''
Description: 
version: v1.0
Author: HTY
Date: 2025-08-13 21:38:14
'''
import math

def point_to_segment_distance(point, seg_start, seg_end):
    x0, y0 = point
    x1, y1 = seg_start
    x2, y2 = seg_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(x0 - x1, y0 - y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)

    if t < 0:
        nearest_x, nearest_y = x1, y1
    elif t > 1:
        nearest_x, nearest_y = x2, y2
    else:
        nearest_x = x1 + t * dx
        nearest_y = y1 + t * dy
    
    return math.hypot(x0 - nearest_x, y0 - nearest_y)