from torch import nn
from torch.nn import functional as F

class Involution(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, stride, groups, reduce_ratio=1, dilation=(1, 1), padding=(3, 3), bias=False):
        super().__init__()
        self.bias = bias
        self.padding = padding
        self.dilation = dilation
        self.reduce_ratio = reduce_ratio
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.output_ch = output_ch
        self.input_ch = input_ch
        self.init_mapping = nn.Conv2d(in_channels=self.input_ch, out_channels=self.output_ch, kernel_size=(1, 1), stride=(1, 1), bias=self.bias) if self.input_ch != self.output_ch else nn.Identity()
        self.reduce_mapping = nn.Conv2d(in_channels=self.input_ch, out_channels=self.output_ch // self.reduce_ratio, kernel_size=(1, 1), stride=(1, 1), bias=self.bias)
        self.span_mapping = nn.Conv2d(in_channels=self.output_ch // self.reduce_ratio, out_channels=self.kernel_size[0] * self.kernel_size[1] * self.groups, kernel_size=(1, 1), stride=(1, 1),
                                      bias=self.bias)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        self.pooling = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        self.sigma = nn.Sequential(
            nn.BatchNorm2d(num_features=self.output_ch // self.reduce_ratio, momentum=0.3), nn.ReLU())

    def forward(self, inputs):
        batch_size, _, in_height, in_width = inputs.shape
        out_height = (in_height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                     // self.stride[0] + 1
        out_width = (in_width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                    // self.stride[1] + 1

        unfolded_inputs = self.unfold(self.init_mapping(inputs))
        inputs = F.adaptive_avg_pool2d(inputs,(out_height,out_width))
        unfolded_inputs = unfolded_inputs.view(batch_size, self.groups, self.output_ch // self.groups, self.kernel_size[0] * self.kernel_size[1], out_height, out_width)

        kernel = self.pooling(self.span_mapping(self.sigma(self.reduce_mapping((inputs)))))
        kernel = kernel.view(batch_size, self.groups, self.kernel_size[0] * self.kernel_size[1], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        output = (kernel * unfolded_inputs).sum(dim=3)

        output = output.view(batch_size, -1, output.shape[-2], output.shape[-1])
        return output
