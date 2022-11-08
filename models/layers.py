import torch
import torch.nn as nn
import math

class PixelNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,x,epsilon=1e-8):
        # ? (batch_size,c,w,h)
        out=x*torch.rsqrt(torch.mean(torch.square(x),dim=1,keepdim=True)+epsilon)
        return out

class MinibatchStd(nn.Module):
    def __init__(self,group_size=4) -> None:
        super().__init__()
        self.group_size=group_size
        # 注意: batch_sizeをgroup_sizeで割り切れる必要がある

    def forward(self,x:torch.Tensor):

        # batch sizeをgroup_sizeで割り切れる必要がある 
        # TODO: assertつける？
        group_size= min(self.group_size,x.size(0))

        bs,c,h,w=x.size() # (batch_size, channel, width, height)
        # channlを分割 channel => group_size x X
        # input shape : [NCHW]
        # output shape: [GMCHW]  C*M=N
        y=x.view(group_size,-1,c,h,w)

        # outpu shape: [GMCHW]
        y = y - torch.mean(y,dim=0,keepdim=True)
        # [MCHW]  groupごとの分散を求める
        y = torch.mean(y.square(),dim=0)
        # [MCHW]  groupごとの標準偏差をもとめる(std)
        y = torch.sqrt(y+1e-8)
        # [M111]  Take average over fmaps and pixels.
        y = torch.mean(y, dim=[1,2,3],keepdim=True)
        #  [N1HW]  Replicate over group and pixels.
        y = torch.tile(y,[group_size,1,h,w])
        # [NCHW]  作成した特徴マップを追加する。
        # 入力のサイズよりchannel数が1つ増える
        y = torch.cat([x,y],dim=1)
        return y


# torch.viewのmodule版
# batch_size以外の部分のshapeを指定する
class View(nn.Module):
    def __init__(self,*shape) -> None:
        super().__init__()
        # shapeはbatch_size以外の部分
        # bach_sizeはそのまま
        self.shape=shape
    
    def forward(self,x):
        return x.view(x.size(0),*self.shape)

# We initialize all bias parameters to zero and all weights according to the normal distribution with
# unit variance. However, we scale the weights with a layer-specific constant at runtime as described
# in Section 4.1.

class EqualizedLinear(nn.Module):
    def __init__(self,in_features, out_features, bias=True, device=None, dtype=None,init_bias_zero=True,use_wscale=True,gain=math.sqrt(2)):
        super().__init__()
        # init_bias_zero: biasの初期値に0を使用する
        # use_wscale:  forwardでweightをscaleする
        assert use_wscale==True

        self.in_features=in_features
        self.out_features=out_features
        self.gain=gain
        self.use_wscale=use_wscale

        self.linear=nn.Linear(in_features,out_features,bias=bias,device=device,dtype=dtype)
        # バイアスをコピー
        self.bias=self.linear.bias
        # linearのbiasをなくす
        self.linear.bias=None

        # self.weight=nn.Parameter(torch.empty(out_features,in_features))
        # self.bias=nn.Parameter(torch.empty(out_features))

        # weightにN(0,1)を設定
        # nn.init.normal_(self.linear.weight)
        nn.init.normal_(self.linear.weight)

        if init_bias_zero:
            # biasに0を設定
            # nn.init.constant_(self.linear.bias,0)
            nn.init.constant_(self.bias,0)
        

        
        # scale weights at runtime
        # per-layer normalization constant from He's initializer (He et al.2015)
        # batch_size,channel=x.size()
        fan_in=in_features # channel
        std= self.gain/math.sqrt(fan_in)

        self.wscale=std

    def forward(self,x):


        if self.use_wscale:


            # self.linear.register_parameter('weight',new_weight)

            # w=self.linear.weight * wscale
            # w=self.linear.weight*self.wscale
            out=self.linear(x*self.wscale)+self.bias
            # out = torch.nn.functional.linear(x,w,self.bias)
            # out=self.linear(x)

        else:
            # out =self.linear(x)
            out=None

        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class EqualizedConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None,
        init_bias_zero=True,use_wscale=True,gain=math.sqrt(2)):
        super().__init__()

        assert use_wscale==True

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size

        self.stride=stride
        self.padding=padding
        self.dilation=1
        self.groups=1
        # TODO: assert kernel >= 1 and kernel % 2 == 1

        self.gain=gain
        self.use_wscale=use_wscale


        # if use_wscale:
        #     # TODO: use_wscale=Trueの時　padding_mode='zeros'以外の処理作る?
        #     assert padding_mode=='zeros'

        # self.conv2d=nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,dilation=dilation,groups=groups,bias=bias,padding_mode=padding_mode,
        #                         device=device,dtype=dtype)

        self.conv2d=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)

        # self.weight=nn.Parameter(torch.empty((out_channels,in_channels,kernel_size[0],kernel_size[1])))
        # self.bias=nn.Parameter(torch.empty(out_channels))

        self.bias=self.conv2d.bias
        self.conv2d.bias=None

        # weightにN(0,1)を設定
        # nn.init.normal_(self.conv2d.weight)
        nn.init.normal_(self.conv2d.weight)
        
        if init_bias_zero:
            # nn.init.constant_(self.conv2d.bias,0)
            nn.init.constant_(self.bias,0)

        # batch_size,channel,height,width=x.size()
        # TODO: kernel_size がtuple出ないとき
        fan_in=in_channels*kernel_size[0]*kernel_size[1]
        print("fain_in conv",(in_channels,kernel_size[0],kernel_size[1]))
        print("fain_in conv",fan_in)
        std= self.gain/math.sqrt(fan_in)
        self.wscale=std

    def forward(self,x):

        # 
        if self.use_wscale:


            # w=self.weight * self.wscale
            # new_weight=self.conv2d.weight.clone()
            # self.conv2d.weight=torch.nn.Parameter(new_weight*wscale)

            # out=self.conv2d(x)
            
            # out= torch.nn.functional.conv2d(x,w,bias=self.bias, stride= self.stride,padding=self.padding,dilation=self.dilation,
            #                                 groups=self.groups)
            out=self.conv2d(x*self.wscale)+self.bias.view(-1,self.bias.size(0),1,1)

        else:
            # out=self.conv2d(x)
            out=None
            assert False

        return out

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        s+=f", padding={self.padding}"
        # if self.padding != (0,) * len(self.padding):
        #     s += ', padding={padding}'
        # if self.dilation != (1,) * len(self.dilation):
        #     s += ', dilation={dilation}'
        # if self.output_padding != (0,) * len(self.output_padding):
        #     s += ', output_padding={output_padding}'
        # if self.groups != 1:
        #     s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        # if self.padding_mode != 'zeros':
        #     s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)