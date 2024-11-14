"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms
from model import Net, NASNetMobile, DINOv2, EfficientNetB3a


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        elif self.model_name == "nasnet_mobile":
            return NASNetMobile() 
        elif self.model_name == "dinov2":
            return DINOv2()
        elif self.model_name == "efficientnet_b3a":
            return EfficientNetB3a()  # Add EfficientNetB3a model
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name in ["basic_cnn", "nasnet_mobile", "dinov2", "efficientnet_b3a"]:
            return data_transforms
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform



