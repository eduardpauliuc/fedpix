from PIL import Image
import os
import matplotlib.pyplot as plt
images_dir = '../generated/handpicked/central_200'
# Load your images
input_image = Image.open(os.path.join(images_dir, "input.png"))
output_image1 = Image.open(os.path.join(images_dir, "output.png"))
output_image2 = Image.open(os.path.join(images_dir, "20_central_50.png"))
output_image3 = Image.open(os.path.join(images_dir, "20_central_100.png"))
output_image4 = Image.open(os.path.join(images_dir, "20_central_150.png"))
output_image5 = Image.open(os.path.join(images_dir, "20_central_200.png"))

# Create a subplot with 1 row and 5 columns
fig, axes = plt.subplots(2, 3, figsize=(9, 6))

# Display the images
axes[0][0].imshow(input_image)
axes[0][0].set_title("Input")
axes[0][0].axis("off")

axes[0][1].imshow(output_image1)
axes[0][1].set_title("Real")
axes[0][1].axis("off")

axes[0][2].imshow(output_image2)
axes[0][2].set_title("50 epochs")
axes[0][2].axis("off")

axes[1][0].imshow(output_image3)
axes[1][0].set_title("100 epochs")
axes[1][0].axis("off")

axes[1][1].imshow(output_image4)
axes[1][1].set_title("150 epochs")
axes[1][1].axis("off")

axes[1][2].imshow(output_image5)
axes[1][2].set_title("200 epochs")
axes[1][2].axis("off")

# Show the plot
plt.tight_layout()
plt.show()