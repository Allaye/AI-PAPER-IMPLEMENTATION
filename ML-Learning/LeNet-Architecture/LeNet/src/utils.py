import matplotlib.pyplot as plt

def plot_image(image):
    """
    plot a single tensor image, the function expect PIL image or a tensor
    """
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.show()
    return None

