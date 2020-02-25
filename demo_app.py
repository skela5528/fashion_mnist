import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import copy

from networks import Net3Conv, Net9Conv
from efficientnet_pytorch import EfficientNet

CONFIG = dict()
CONFIG["model_path"] = \
    "resources/model-Net9Conv_Feb24_1952_epochs-050_acc-0.942_params-0154K_t-053.3_exp-Conv9_LR0.10000_aug-True.pth"
CONFIG["model_name"] = "Net9Conv"  # "EfficientNet_b0"
CONFIG["images_dir"] = "images_16/"
CONFIG["model_input_size"] = 28  # 32 for EfficientNet_b0
CONFIG["out_dir"] = "../"
CONFIG["classes"] = ['T-shirt/top',
                     'Trouser',
                     'Pullover',
                     'Dress',
                     'Coat',
                     'Sandal',
                     'Shirt',
                     'Sneaker',
                     'Bag',
                     'Ankle boot']
CONFIG["num_classes"] = 10
CONFIG["in_channels"] = 1
CONFIG["batch_size"] = 32
CONFIG["demo_size"] = 500
CONFIG["demo_fps"] = 2


def plot_tensor_image(batch_imgs: torch.Tensor, labels=None):
    n = CONFIG["batch_size"]
    batch_imgs = batch_imgs.squeeze().detach().cpu()
    fig, axs = plt.subplots(n, 1)
    for i in range(n):
        title = labels[i] if labels is not None else ""
        axs[i].imshow(batch_imgs[i],  cmap="gray")
        axs[i].set_title(title)
        axs[i].axis("off")
    fig.tight_layout()
    plt.show()


class IO:
    """Video related functions."""

    FONT = ImageFont.truetype("resources/UbuntuMono-R.ttf", 18)

    def __init__(self, out_dir, video_size, fps):
        time_stamp = time.strftime("%b%d_%H%M", time.localtime(time.time()))
        self.video_path = os.path.join(out_dir, "video_{}.mp4".format(time_stamp))
        self.video_dims = (video_size, video_size)
        self.fps = fps

    def get_cv2_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, self.fps, self.video_dims)
        return out

    @staticmethod
    def cv2_video_append_frame(video_writer, frame):
        arr = np.asarray(frame)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        del frame
        video_writer.write(arr)

    @staticmethod
    def draw_prediction_info(draw: ImageDraw, label, score, model_name):
        text = "{:12s} - {:4.2f}".format(label, score)
        draw.text(xy=(10, 10), text=text, fill="lime", font=IO.FONT)
        draw.text(xy=(10, 30), text="Model input", fill="lime", font=IO.FONT)
        draw.text(xy=(10, 50), text="Model - {}".format(model_name), fill="lime", font=IO.FONT)

    @staticmethod
    def draw_model_input(image, model_input: np.array):
        position = (130, 27)
        model_input = cv2.cvtColor(model_input, 8)
        model_input = (model_input * 255).astype(np.uint8)
        model_input = Image.fromarray(model_input)
        image.paste(model_input, position)  # type Image


class ImgsDirDataset(Dataset):
    """Dataset for loading single directory with images. Suitable only for inference."""

    def __init__(self, images_dir, model_input_size, demo_img_size):
        self.images_dir = images_dir
        self.size = model_input_size
        self.demo_img_size = demo_img_size
        self.image_path = self.__get_image_paths()
        self.preprocess = self.__init_network_preprocess()
        self.demo_resize = self.__init_demo_preprocess()

    def __get_image_paths(self, img_ext=("png", "jpeg", "jpg")):
        assert os.path.exists(self.images_dir), "Images dir doesn't exist!!"
        file_names = os.listdir(self.images_dir)
        image_names = [n for n in file_names if n.split(".")[-1] in img_ext]
        image_names.sort()
        np.random.seed(1)
        image_names_shuffled = np.random.choice(image_names, size=len(image_names), replace=False)
        image_paths = [os.path.join(self.images_dir, n) for n in image_names_shuffled]
        print(" - {} images found... ".format(len(image_names)))
        return image_paths

    def __init_network_preprocess(self):
        """Preprocess pipeline."""
        transforms_list = list()
        transforms_list.append(transforms.Resize(self.size, interpolation=Image.BICUBIC))
        transforms_list.append(transforms.CenterCrop(self.size))
        transforms_list.append(transforms.Grayscale(num_output_channels=1))
        # transforms_list.append(transforms.RandomHorizontalFlip())
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Lambda(lambda x: 1 - x))
        return transforms.Compose(transforms_list)

    def __init_demo_preprocess(self):
        """Preprocess pipeline."""
        transforms_list = list()
        transforms_list.append(transforms.Resize(self.demo_img_size, interpolation=Image.BICUBIC))
        transforms_list.append(transforms.CenterCrop(self.demo_img_size))
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self.image_path[idx])
        model_input = self.preprocess(image)
        demo_img_numpy = np.array(self.demo_resize(image))
        return model_input, demo_img_numpy


class DemoApp:
    supported_model_names = ["EfficientNet_b0", "Net3Conv", "Net9Conv"]

    def __init__(self, config):
        self.config = copy.deepcopy(config)

    def __get_dataloader(self):
        args = [self.config["images_dir"], self.config["model_input_size"], self.config["demo_size"]]
        dataset = ImgsDirDataset(*args)
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"])
        return dataloader

    def __init_model_instance(self, model_name) -> nn.Module:
        assert model_name in self.supported_model_names, "{} model is not supported!!".format(model_name)
        model_instance = None
        if model_name == "Net3Conv":
            model_instance = Net3Conv()
        elif model_name == "Net9Conv":
            model_instance = Net9Conv()
        elif model_name == "EfficientNet_b0":
            model_instance = EfficientNet.from_pretrained(model_name="efficientnet-b0",
                                                          num_classes=self.config["num_classes"],
                                                          in_channels=self.config["in_channels"])
        return model_instance.cuda()

    def __load_model(self) -> nn.Module:
        assert os.path.exists(self.config["model_path"]), \
            "Model doesn't exist - {}!!".format(self.config["model_path"])
        model_instance = self.__init_model_instance(self.config["model_name"])
        model_dict = torch.load(self.config["model_path"])
        model_instance.load_state_dict(model_dict)
        model_instance.eval()
        print(" - {} model is loaded from {}".
              format(model_instance.__class__.__name__, self.config["model_path"]))
        return model_instance

    def __get_predictions_from_logit_scores(self, pred_logit_scores):
        probs = nn.functional.softmax(pred_logit_scores, dim=1)
        probs = probs.detach().cpu().numpy().tolist()
        probs_max = [round(max(p), 5) for p in probs]
        predicted_classes = pred_logit_scores.argmax(dim=1)
        predicted_classes = predicted_classes.detach().cpu()
        predicted_labels = [self.config["classes"][cl] for cl in predicted_classes]
        predictions_with_score = list(zip(predicted_labels, probs_max))
        return predictions_with_score

    def process_images_dir(self):
        """Main function."""
        data = self.__get_dataloader()
        model = self.__load_model()
        io = IO(self.config["out_dir"], self.config["demo_size"], self.config["demo_fps"])
        video_writer = io.get_cv2_video_writer()
        n_batches = len(data)
        for bid, (batch_imgs, imgs) in enumerate(data):
            print(" - Processing batch {:4d} out of {:4d} [{:2.0f}%]".format(bid, n_batches, bid/n_batches*100))
            pred_probs = model(batch_imgs.cuda())
            pred_classes = self.__get_predictions_from_logit_scores(pred_probs)
            # plot_tensor_image(batch_imgs, pred_classes)
            batch_imgs = batch_imgs.detach().cpu().numpy()
            for i in range(len(imgs)):
                img_i = imgs[i].detach().cpu().numpy()
                img_i = Image.fromarray(img_i)
                io.draw_model_input(img_i, *batch_imgs[i])
                draw_i = ImageDraw.Draw(img_i)
                io.draw_prediction_info(draw_i, *pred_classes[i], self.config["model_name"])
                io.cv2_video_append_frame(video_writer, img_i)
        print(" - Done!!! Video saved here: {}".format(io.video_path))


if __name__ == '__main__':
    demo = DemoApp(CONFIG)
    demo.process_images_dir()
