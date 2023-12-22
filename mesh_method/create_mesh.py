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
        self.image = Image.open(self.image_path)  # Replace with the actual path to your JPEG image
        self.photo = ImageTk.PhotoImage(self.image)

        # Add the image to the canvas at coordinates (0, 0)

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
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

            # # Draw the extended lines
            self.canvas.create_line(intersection1[0], intersection1[1], intersection2[0], intersection2[1], fill="blue",
                                    tags="mesh")
            self.canvas.create_line(intersection2[0], intersection2[1], intersection3[0], intersection3[1], fill="blue",
                                    tags="mesh")
            self.canvas.create_line(intersection3[0], intersection3[1], intersection4[0], intersection4[1], fill="blue",
                                    tags="mesh")
            self.canvas.create_line(intersection4[0], intersection4[1], intersection1[0], intersection1[1], fill="blue",
                                    tags="mesh")

            # # Draw the extended lines if valid intersection points are available
            # if intersection1 is not None:
            #     self.draw_extended_line(x1, y1, intersection1[0], intersection1[1], x2 - x1, y2 - y1)
            # if intersection2 is not None:
            #     self.draw_extended_line(x2, y2, intersection2[0], intersection2[1], x2 - x1, y2 - y1)
            # if intersection3 is not None:
            #     self.draw_extended_line(x2, y2, intersection3[0], intersection3[1], x2 - x1, y2 - y1)
            # if intersection4 is not None:
            #     self.draw_extended_line(x1, y1, intersection4[0], intersection4[1], x2 - x1, y2 - y1)

    def draw_extended_line(self, x1, y1, x2, y2, direction_x, direction_y):
        # Calculate the perpendicular vector
        perpendicular_dx = -direction_y
        perpendicular_dy = direction_x

        # Extend the lines beyond the canvas borders
        extended_line_start = (x1 - 10 * perpendicular_dx, y1 - 10 * perpendicular_dy)
        extended_line_end = (x2 + 10 * perpendicular_dx, y2 + 10 * perpendicular_dy)

        self.canvas.create_line(extended_line_start, extended_line_end, fill="red", tags="mesh")

    def calculate_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return x, y

    def drag_point(self, event):
        for i, point in enumerate(self.mesh_points):
            x, y = point
            distance = np.sqrt((event.x - x)**2 + (event.y - y)**2)
            if distance < 30:  # Check if the mouse is within a certain distance of the point
                self.mesh_points[i] = [event.x, event.y]
                self.draw_mesh()
                break

    def run(self):
        self.root.mainloop()

def main():
    root = tk.Tk()
    image_path = "img.jpg"  # Replace with your image path

    editor = ImageMeshEditor(root, image_path)
    editor.run()

if __name__ == "__main__":
    main()
