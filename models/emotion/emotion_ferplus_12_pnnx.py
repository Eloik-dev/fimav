import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_0 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=1, kernel_size=(3,3), out_channels=64, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_1 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=64, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_2 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=128, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_3 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=128, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_4 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=128, kernel_size=(3,3), out_channels=256, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_5 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_6 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_7 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_8 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding='same', padding_mode='zeros', stride=(1,1))
        self.conv2d_9 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=256, kernel_size=(3,3), out_channels=256, padding='same', padding_mode='zeros', stride=(1,1))
        self.linear_0 = nn.Linear(bias=True, in_features=4096, out_features=1024)
        self.linear_1 = nn.Linear(bias=True, in_features=1024, out_features=1024)
        self.linear_2 = nn.Linear(bias=True, in_features=1024, out_features=8)

        archive = zipfile.ZipFile('./models/emotion/emotion_ferplus_12.pnnx.bin', 'r')
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.weight', (64,1,3,3), 'float32')
        self.conv2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_1.weight', (64,64,3,3), 'float32')
        self.conv2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_2.weight', (128,64,3,3), 'float32')
        self.conv2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_3.weight', (128,128,3,3), 'float32')
        self.conv2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_4.weight', (256,128,3,3), 'float32')
        self.conv2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_5.weight', (256,256,3,3), 'float32')
        self.conv2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_6.weight', (256,256,3,3), 'float32')
        self.conv2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_7.weight', (256,256,3,3), 'float32')
        self.conv2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_8.weight', (256,256,3,3), 'float32')
        self.conv2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_9.weight', (256,256,3,3), 'float32')
        self.linear_0.bias = self.load_pnnx_bin_as_parameter(archive, 'linear_0.bias', (1024), 'float32')
        self.linear_0.weight = self.load_pnnx_bin_as_parameter(archive, 'linear_0.weight', (1024,4096), 'float32')
        self.linear_1.bias = self.load_pnnx_bin_as_parameter(archive, 'linear_1.bias', (1024), 'float32')
        self.linear_1.weight = self.load_pnnx_bin_as_parameter(archive, 'linear_1.weight', (1024,1024), 'float32')
        self.linear_2.bias = self.load_pnnx_bin_as_parameter(archive, 'linear_2.bias', (8), 'float32')
        self.linear_2.weight = self.load_pnnx_bin_as_parameter(archive, 'linear_2.weight', (8,1024), 'float32')
        self.Parameter4_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter4.data', (64,1,1,), 'float32')
        self.Parameter24_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter24.data', (64,1,1,), 'float32')
        self.Parameter64_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter64.data', (128,1,1,), 'float32')
        self.Parameter84_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter84.data', (128,1,1,), 'float32')
        self.Parameter576_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter576.data', (256,1,1,), 'float32')
        self.Parameter596_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter596.data', (256,1,1,), 'float32')
        self.Parameter616_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter616.data', (256,1,1,), 'float32')
        self.Parameter656_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter656.data', (256,1,1,), 'float32')
        self.Parameter676_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter676.data', (256,1,1,), 'float32')
        self.Parameter696_data = self.load_pnnx_bin_as_parameter(archive, 'Parameter696.data', (256,1,1,), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = ((v_0 - 127.500000) / 255.000000)
        v_2 = self.conv2d_0(v_1)
        v_3 = self.Parameter4_data
        v_4 = (v_2 + v_3)
        v_5 = F.relu(input=v_4)
        v_6 = self.conv2d_1(v_5)
        v_7 = self.Parameter24_data
        v_8 = (v_6 + v_7)
        v_9 = F.relu(input=v_8)
        v_10 = F.max_pool2d(input=v_9, ceil_mode=False, kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        v_11 = self.conv2d_2(v_10)
        v_12 = self.Parameter64_data
        v_13 = (v_11 + v_12)
        v_14 = F.relu(input=v_13)
        v_15 = self.conv2d_3(v_14)
        v_16 = self.Parameter84_data
        v_17 = (v_15 + v_16)
        v_18 = F.relu(input=v_17)
        v_19 = F.max_pool2d(input=v_18, ceil_mode=False, kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        v_20 = self.conv2d_4(v_19)
        v_21 = self.Parameter576_data
        v_22 = (v_20 + v_21)
        v_23 = F.relu(input=v_22)
        v_24 = self.conv2d_5(v_23)
        v_25 = self.Parameter596_data
        v_26 = (v_24 + v_25)
        v_27 = F.relu(input=v_26)
        v_28 = self.conv2d_6(v_27)
        v_29 = self.Parameter616_data
        v_30 = (v_28 + v_29)
        v_31 = F.relu(input=v_30)
        v_32 = F.max_pool2d(input=v_31, ceil_mode=False, kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        v_33 = self.conv2d_7(v_32)
        v_34 = self.Parameter656_data
        v_35 = (v_33 + v_34)
        v_36 = F.relu(input=v_35)
        v_37 = self.conv2d_8(v_36)
        v_38 = self.Parameter676_data
        v_39 = (v_37 + v_38)
        v_40 = F.relu(input=v_39)
        v_41 = self.conv2d_9(v_40)
        v_42 = self.Parameter696_data
        v_43 = (v_41 + v_42)
        v_44 = F.relu(input=v_43)
        v_45 = F.max_pool2d(input=v_44, ceil_mode=False, kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        v_46 = v_45.reshape(1, 4096)
        v_47 = self.linear_0(v_46)
        v_48 = F.relu(input=v_47)
        v_49 = self.linear_1(v_48)
        v_50 = F.relu(input=v_49)
        v_51 = self.linear_2(v_50)
        return v_51

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 64, 64, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("./models/emotion/emotion_ferplus_12_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 64, 64, dtype=torch.float)

    torch.onnx.export(net, v_0, "./models/emotion/emotion_ferplus_12_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 1, 64, 64, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
