import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub import notebook_login, push_to_hub_keras, HfApi, login
import tf2onnx
from transformers import PreTrainedModel, PretrainedConfig, TFAutoModel, AutoConfig
from onnx2pytorch import ConvertModel
import onnx
import tensorflow as tf
token = "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"  #Now deleted

login(token=token)

# Wrap PyTorch model in nn.Module
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)


class CustomModelConfig(PretrainedConfig):
    model_type = "custom_model"
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

# Define Hugging Face PyTorch Model
class CustomHFModel(PreTrainedModel):
    config_class = CustomModelConfig
    def __init__(self, config, model):
        super().__init__(config)
        self.model = model

    def forward(self, x):
        return self.model(x)
keras_model = tf.keras.models.load_model(r".\models\Teacher.h5")

onnx_model_path = r".\models\model.onnx"
onnx_model, _ = tf2onnx.convert.from_keras(keras_model, output_path=onnx_model_path)

# Convert ONNX to PyTorch
onnx_model = onnx.load(onnx_model_path)
# torch.onnx.export(onnx_model, torch.randn(1, 3, 224, 224), "model.pth")
pytorch_model = ConvertModel(onnx_model)

pytorch_model = WrappedModel(pytorch_model)

torch.save(pytorch_model.state_dict(), r".\models\model.pth")


def upload_pytorch_model():
    pytorch_model.load_state_dict(torch.load(r".\models\model.pth"))
    config = CustomModelConfig()
    model = CustomHFModel(config, pytorch_model)
    model.push_to_hub("adityaroy10/SpineNet49S")
    print("PyTorch Model uploaded successfully!")


upload_pytorch_model()


