import cv2
import numpy as np
import tkinter as tk

class ImageMeshEditor:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Image Mesh Editor")

        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.height, self.width, _ = self.image.shape

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()

        self.mesh_points = np.array([
            [100, 100],
            [400, 100],
            [400, 300],
            [100, 300]
        ], dtype=np.float32)

        self.draw_mesh()

        self.canvas.bind("<B1-Motion>", self.drag_point)

    def draw_mesh(self):
        self.canvas.delete("mesh")

        for point in self.mesh_points:
            x, y = point
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", tags="mesh")

        # Draw lines extending to infinity
        for i in range(4):
            x1, y1 = self.mesh_points[i]
            x2, y2 = self.mesh_points[(i + 1) % 4]

            # Calculate intersections with image borders
            intersection1 = self.calculate_intersection(x1, y1, x2, y2, 0, 0, self.width, 0)
            intersection2 = self.calculate_intersection(x1, y1, x2, y2, self.width, 0, self.width, self.height)
            intersection3 = self.calculate_intersection(x1, y1, x2, y2, self.width, self.height, 0, self.height)
            intersection4 = self.calculate_intersection(x1, y1, x2, y2, 0, self.height, 0, 0)

            # Draw the extended lines if valid intersection points are available
            if intersection1 is not None:
                self.draw_extended_line(x1, y1, intersection1[0], intersection1[1])
            if intersection2 is not None:
                self.draw_extended_line(x2, y2, intersection2[0], intersection2[1])
            if intersection3 is not None:
                self.draw_extended_line(x2, y2, intersection3[0], intersection3[1])
            if intersection4 is not None:
                self.draw_extended_line(x1, y1, intersection4[0], intersection4[1])

    def draw_extended_line(self, x1, y1, x2, y2):
        # Draw a line between two points
        self.canvas.create_line(x1, y1, x2, y2, fill="blue", tags="mesh")

        # Calculate the direction vector of the line
        dx, dy = x2 - x1, y2 - y1

        # Calculate the perpendicular vector
        perpendicular_dx = -dy
        perpendicular_dy = dx

        # Extend the lines beyond the canvas borders
        extended_line_start = np.array([x1 - 10 * perpendicular_dx, y1 - 10 * perpendicular_dy])
        extended_line_end = np.array([x2 + 10 * perpendicular_dx, y2 + 10 * perpendicular_dy])

        # Find the intersection point using least squares method
        intersection = self.calculate_intersection_least_squares(extended_line_start, extended_line_end)

        if intersection is not None:
            self.canvas.create_line(extended_line_start[0], extended_line_start[1],
                                    intersection[0], intersection[1], fill="red", tags="mesh")
            self.canvas.create_line(extended_line_end[0], extended_line_end[1],
                                    intersection[0], intersection[1], fill="red", tags="mesh")

    def calculate_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # Calculate the intersection point of two lines
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Check if the lines are parallel (denominator is zero)
        if denominator == 0:
            return None

        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return x, y

    def calculate_intersection_least_squares(self, point1, point2):
        # Find the intersection point using least squares method
        A = np.vstack([point1, point2, [1, 1]])
        b = np.array([point1[0] * point1[1], point2[0] * point2[1], 1])
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return result[:2]

    def drag_point(self, event):
        for i, point in enumerate(self.mesh_points):
            x, y = point
            distance = np.sqrt((event.x - x)**2 + (event.y - y)**2)
            if distance < 10:  # Check if the mouse is within a certain distance of the point
                self.mesh_points[i] = [event.x, event.y]
                self.draw_mesh()
                break

    def run(self):
        self.root.mainloop()

def main():
    root = tk.Tk()
    image_path = "image.jpg"  # Replace with your image path
    editor = ImageMeshEditor(root, image_path)
    editor.run()

if __name__ == "__main__":
    main()
