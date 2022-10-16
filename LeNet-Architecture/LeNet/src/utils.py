import matplotlib.pyplot as plt
import torch

def plot_image(image):
    """
    plot a single tensor image, the function expect PIL image or a tensor
    """
    plt.imshow(image.permute(1, 2, 0), cmap='gray')
    plt.show()
    return None


def save_checkpoint(model, epoch, optimizer, best_accuracy) -> None:
    """
    save the trained model on each checkpoint
    :param model:
    :param epoch:
    :param optimizer:
    :param best_accuracy:
    :return: None type
    """
    check_point = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy
    }
    torch.save(check_point, f"/models/checkpoint{epoch}.pth")
    print(f"checkpoint successfully saved at epoch {epoch}")
    return None
