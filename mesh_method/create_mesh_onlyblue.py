import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

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
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height]
        ], dtype=np.float32)

        self.draw_mesh()

        self.canvas.bind("<B1-Motion>", self.drag_point)

    def draw_mesh(self):
        self.canvas.delete("mesh")

        for point in self.mesh_points:
            x, y = point
            self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="red", tags="mesh")

        # Draw original lines extending to infinity
        for i in range(4):
            x1, y1 = self.mesh_points[i]
            x2, y2 = self.mesh_points[(i + 1) % 4]

            # Calculate intersections with image borders
            intersection1 = self.calculate_intersection(x1, y1, x2, y2, 0, 0, self.width, 0)
            intersection2 = self.calculate_intersection(x1, y1, x2, y2, self.width, 0, self.width, self.height)
            intersection3 = self.calculate_intersection(x1, y1, x2, y2, self.width, self.height, 0, self.height)
            intersection4 = self.calculate_intersection(x1, y1, x2, y2, 0, self.height, 0, 0)

            # Draw the extended lines
            self.canvas.create_line(intersection1[0], intersection1[1], intersection2[0], intersection2[1], fill="blue", tags="mesh")
            self.canvas.create_line(intersection2[0], intersection2[1], intersection3[0], intersection3[1], fill="blue", tags="mesh")
            self.canvas.create_line(intersection3[0], intersection3[1], intersection4[0], intersection4[1], fill="blue", tags="mesh")
            self.canvas.create_line(intersection4[0], intersection4[1], intersection1[0], intersection1[1], fill="blue", tags="mesh")

            # Draw three parallel lines
            for j in range(1, 4):
                delta_x = (x2 - x1) / 4 * j
                delta_y = (y2 - y1) / 4 * j

                parallel_intersection1 = self.calculate_intersection(x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y, 0, 0, self.width, 0)
                parallel_intersection2 = self.calculate_intersection(x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y, self.width, 0, self.width, self.height)
                parallel_intersection3 = self.calculate_intersection(x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y, self.width, self.height, 0, self.height)
                parallel_intersection4 = self.calculate_intersection(x1 + delta_x, y1 + delta_y, x2 + delta_x, y2 + delta_y, 0, self.height, 0, 0)

                self.canvas.create_line(parallel_intersection1[0], parallel_intersection1[1], parallel_intersection2[0], parallel_intersection2[1], fill="blue", tags="mesh")
                self.canvas.create_line(parallel_intersection2[0], parallel_intersection2[1], parallel_intersection3[0], parallel_intersection3[1], fill="blue", tags="mesh")
                self.canvas.create_line(parallel_intersection3[0], parallel_intersection3[1], parallel_intersection4[0], parallel_intersection4[1], fill="blue", tags="mesh")
                self.canvas.create_line(parallel_intersection4[0], parallel_intersection4[1], parallel_intersection1[0], parallel_intersection1[1], fill="blue", tags="mesh")


    def drag_point(self, event):
        for i, point in enumerate(self.mesh_points):
            x, y = point
            distance = np.sqrt((event.x - x)**2 + (event.y - y)**2)
            if distance < 10:
                self.mesh_points[i] = [event.x, event.y]
                self.draw_mesh()
                break

    def calculate_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return x, y

    def calculate_parallel_line(self, x1, y1, x2, y2, shift):
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        x_shifted1 = x1 + shift * (y2 - y1) / length
        y_shifted1 = y1 - shift * (x2 - x1) / length
        x_shifted2 = x2 + shift * (y2 - y1) / length
        y_shifted2 = y2 - shift * (x2 - x1) / length
        return x_shifted1, y_shifted1, x_shifted2, y_shifted2

    def run(self):
        self.root.mainloop()

def main():
    root = tk.Tk()
    image_path = "image.png"  # Replace with your image path
    editor = ImageMeshEditor(root, image_path)
    editor.run()

if __name__ == "__main__":
    main()
