import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetContractingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 64, kernel_size = 3, padding = 0)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 0)

    def forward(self, x):
        print(f"Input shape : {tuple(x.shape)}")

        x = self.conv1(x)
        print(f"After 1st 3x3 convolution : {tuple(x.shape)}")
        x = F.relu(x)

        x = self.conv2(x)
        print(f"After 2nd 3x3 convolution : {tuple(x.shape)}")
        x = F.relu(x)

        return x

def main() -> None:
    block = UNetContractingBlock()

    input = torch.randn(1,1,572, 572)
    otuput = block(input)

if __name__ == "__main__":
    main()
