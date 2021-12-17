import matplotlib.pyplot as plt
from dataset_loader import prepare_test_set

def plot_image(image):
    """
    plot a single tensor image
    """
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.show()
    return None

