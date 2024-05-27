
import numpy as np
import matplotlib.pyplot as plt

IMGS_PATH = "imgs"

def save_image(x, label="img", layer=0):
    x = x.squeeze().detach().cpu().numpy()
    x = x[layer]
    plt.imsave(f"{IMGS_PATH}/{label}.png", x)

def get_images(model, dataset, number_of_images=10):
    model.eval()
    for i in range(number_of_images):
        x, y = dataset[i]
        x = x.unsqueeze(0)
        y_hat = model(x)
        save_image(x, f"input_{i}", layer=1) # 0 is masked
        save_image(y, f"target_{i}")
        save_image(y_hat, f"output_{i}")
        print(f"Image {i} saved") 


def save_losses(trainer, model_name):
    plt.plot(trainer.train_losses, label="train")
    plt.plot(trainer.val_losses, label="val")
    plt.savefig(f"{model_name}_losses.png")