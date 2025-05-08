import math
import matplotlib.pyplot as plt

def circle_rectangle_intersections(rect_corners, circle_center, radius):
    def point_on_segment(px, py, x1, y1, x2, y2):
        return (min(x1, x2) <= px <= max(x1, x2)) and (min(y1, y2) <= py <= max(y1, y2))

    def intersect_circle_with_segment(x1, y1, x2, y2, cx, cy, r):
        dx = x2 - x1
        dy = y2 - y1

        fx = x1 - cx
        fy = y1 - cy

        a = dx ** 2 + dy ** 2
        b = 2 * (fx * dx + fy * dy)
        c = fx ** 2 + fy ** 2 - r ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return []

        discriminant = math.sqrt(discriminant)
        intersections = []
        for sign in [-1, 1]:
            t = (-b + sign * discriminant) / (2 * a)
            if 0 <= t <= 1:
                ix = x1 + t * dx
                iy = y1 + t * dy
                intersections.append((ix, iy))
        return intersections

    rect_edges = []
    if len(rect_corners) != 4:
        raise ValueError("Rectangle must have 4 corners.")

    for i in range(4):
        p1 = rect_corners[i]
        p2 = rect_corners[(i + 1) % 4]
        rect_edges.append((p1, p2))

    cx, cy = circle_center
    r = radius
    intersection_points = []

    for (x1, y1), (x2, y2) in rect_edges:
        pts = intersect_circle_with_segment(x1, y1, x2, y2, cx, cy, r)
        intersection_points.extend(pts)

    # # --- Plotting ---
    # fig, ax = plt.subplots()
    #
    # # Plot rectangle
    # rect_corners_cycle = rect_corners + [rect_corners[0]]  # close the rectangle
    # xs, ys = zip(*rect_corners_cycle)
    # ax.plot(xs, ys, 'k-', label='Rectangle')
    #
    # # Plot circle
    # circle_plot = plt.Circle(circle_center, radius, color='b', fill=False, label='Circle')
    # ax.add_patch(circle_plot)
    #
    # # Plot intersection points
    # if intersection_points:
    #     ix, iy = zip(*intersection_points)
    #     ax.plot(ix, iy, 'ro', label='Intersections')
    #
    # # Formatting
    # ax.set_aspect('equal')
    # ax.grid(True)
    # ax.legend()
    # plt.title("Circle-Rectangle Intersections")
    # plt.show()

    return intersection_points