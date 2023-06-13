from PIL import Image
import os


def split_image(image_path, output_folder):
    img = Image.open(image_path)
    width, height = img.size

    # Ensure the image is the correct size
    assert width == 1200 and height == 600, "Image size must be 1200x600px"

    # Split the image
    img_left = img.crop((0, 0, width // 2, height))
    img_right = img.crop((width // 2, 0, width, height))

    # Save the images
    img_right.save(os.path.join(output_folder, 'input.png'))
    img_left.save(os.path.join(output_folder, 'output.png'))


# Test the function
if __name__ == "__main__":
    split_image("../996.jpg", "../")
