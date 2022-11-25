"""
Created on Fri Nov 26 02:02:55 2021

@author: shubh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class attention2d(nn.Module):
    def __init__(self, in_planes, out_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        
        # keep temp as 3N + 1
        assert temperature%3==1
        
        self.avgpool = nn.AdaptiveAvgPool2d(10)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, kernel_size = 3, stride = 2, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)     
        self.fc2 = nn.Conv2d(hidden_planes, K, kernel_size = 3, stride = 1, bias=True)
        
        if init_weight:
            self._initialize_weights()



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.tanh(x)  # batch_size, K ,2, 2



class Dynamic_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=False, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_channels%groups==0
        self.in_planes = in_channels
        self.out_planes = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_channels,out_channels, ratio, K, temperature)
        self.weight = nn.Parameter(torch.randn(K, out_channels, in_channels//groups, kernel_size+1, kernel_size+1), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_channels))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
            

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])




    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        
        attention_weights = self.attention(x) #batch_size, K ,2 , 2
        
        batch_size, in_planes, height, width = x.size() 
        x = x.view(1, -1, height, width)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        
        K, out_planes, in_planes, (kernel_size), (kernel_size) =  self.weight.size()
        kernel_size = kernel_size-1
        
        weights = self.weight.view(out_planes*in_planes, K,  (kernel_size+1),  (kernel_size+1))
        agg_weights = F.conv2d(weights, attention_weights)
        aggregate_weight = agg_weights.view(batch_size* out_planes, in_planes, kernel_size, kernel_size)
        
        if self.bias:
            # aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            # output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              # dilation=self.dilation, groups=self.groups*batch_size)
            pass
        
        else:  
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output



if __name__ == '__main__':
    x = torch.randn(24, 3,  20)
    model = Dynamic_conv2d(in_planes=3, out_planes=16, kernel_size=3, ratio=0.25, padding=1,)
    x = x.to('cuda:0')
    model.to('cuda')
    # model.attention.cuda()
    # nn.Conv3d()
    print(model(x).shape)
    model.update_temperature()
    
