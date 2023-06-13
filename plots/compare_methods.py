from PIL import Image
import os
import matplotlib.pyplot as plt
images_dir = '../generated/handpicked/1'
# Load your images
input_image = Image.open(os.path.join(images_dir, "input.png"))
output_image1 = Image.open(os.path.join(images_dir, "output.png"))
# output_image2 = Image.open(os.path.join(images_dir, "20_central_50.png"))
output_image3 = Image.open(os.path.join(images_dir, "100.png"))
# output_image4 = Image.open(os.path.join(images_dir, "20_central_150.png"))
output_image5 = Image.open(os.path.join(images_dir, "200.png"))

# Create a subplot with 1 row and 5 columns
fig, axes = plt.subplots(1, 4, figsize=(9, 3))

# Display the images
axes[0].imshow(input_image)
axes[0].set_title("Input")
axes[0].axis("off")

axes[1].imshow(output_image1)
axes[1].set_title("Real")
axes[1].axis("off")

# axes[1][0].imshow(output_image2)
# axes[1][0].set_title("50 epochs")
# axes[1][0].axis("off")

axes[2].imshow(output_image3)
axes[2].set_title("10 rounds")
axes[2].axis("off")

# axes[1][2].imshow(output_image4)
# axes[1][2].set_title("150 epochs")
# axes[1][2].axis("off")

axes[3].imshow(output_image5)
axes[3].set_title("20 rounds")
axes[3].axis("off")

# Show the plot
plt.tight_layout()
plt.show()