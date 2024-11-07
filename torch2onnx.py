from mobilenet.UNet_MobileNet import UNet
import torch
import onnx
import torch.nn as nn

model_path = "model7/MobileNet_UNet_epoch11.pt"

if __name__ == "__main__":
    # input shape尽量选择能被2整除的输入大小
    dummy_input = torch.randn(1, 3, 256, 256)
    # [1] create network
    model = UNet(n_channels=3, num_classes=3)
    model_dict = model.state_dict()
    model.eval()
    print("create U-Net model finised ...")
    # [2] 加载权重
    state_dict = torch.load(model_path,map_location=torch.device('cpu'))
    # model.load_state_dict(state_dict)
    print("load weight to model finised ...")

    # 筛除不加载的层结构
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 更新当前网络的结构字典
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # convert torch format to onnx
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(model,
                      dummy_input,
                      "unet_deconv_cat_dog.onnx",
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=11)
    print("convert torch format model to onnx ...")
    # [4] confirm the onnx file
    net = onnx.load("unet_deconv_cat_dog.onnx")
    # check that the IR is well formed
    onnx.checker.check_model(net)
    # print a human readable representation of the graph
    onnx.helper.printable_graph(net.graph)
