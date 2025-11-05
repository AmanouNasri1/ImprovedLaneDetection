import os
import cv2

def show_image(image_path):
    """Reads an image from the specified path.
    Args:       
        image_path (str): The path to the image file."""
    
    image = cv2.imread(image_path)

    cv2.imshow("Image", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bdd100k_image_path = os.path.join(parent_dir, "bdd100k", "bdd100k", "images","100k", "test", "cb2fe290-8786cd14.jpg")
    
    show_image(bdd100k_image_path)

if __name__ == "__main__":
    main()




    