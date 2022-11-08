import torch
import torch.nn as nn
import numpy as np
import math
from .layers import PixelNorm,View,MinibatchStd,EqualizedConv2d,EqualizedLinear

def num_features(stage:int):
    fmap_base=8192
    fmap_decay=1.0
    fmap_max=512
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


class Generator(nn.Module):
    def __init__(self,resolution=1024,start_resolution=4,alpha=0.0,delta_alpha=0.001,use_wscale=True) -> None:
        super().__init__()

        resolution_log2=int(np.log2(resolution))
        self.resolution=resolution
        assert 2**resolution_log2==resolution, "resolutionは2^xの値を設定する必要あり(e.g. 2^10=1024)"
        self.resolution_log2=resolution_log2

        assert 2**int(np.log2(start_resolution))==start_resolution
        self.current_resolution=start_resolution
        self.current_resolution_log2=int(np.log2(start_resolution))

        self.use_wscale=use_wscale

        self.normalize_latent_vector=PixelNorm()

        activation=nn.LeakyReLU(0.2)
        self.activation=activation


        self.first_block=nn.Sequential(
            # latentが入力
            EqualizedLinear(in_features=512,out_features=512*4*4,gain=math.sqrt(2)/4), # TODO: サイズを変数に変更

            # (batch_size, -1 , 4,4)になるように変形
            View(-1,4,4),
            activation,
            self.norm(),

            # TODO: num_features(1)でok (2) ?確認する
            EqualizedConv2d(in_channels=512,out_channels=num_features(1),kernel_size=(3,3),padding=1),
            self.activation,
            self.norm(),
        )

        # 2(4x4)は上で作成している
        # [3,resolution_log2]の範囲
        self.blocks=nn.ModuleDict({
            f"block_{res_log2}": self.make_block(num_features(res_log2-2),num_features(res_log2-1))  for res_log2 in range(3,resolution_log2+1)
        })

        # [2,resolution_log2]
        # 2はblockを通さないとき。4x4の時
        self.to_rgb_layers=nn.ModuleDict({
            f"to_rgb_{res_log2}" : EqualizedConv2d(in_channels=num_features(res_log2-1),out_channels=3,kernel_size=(1,1),gain=1) for res_log2 in range(2,resolution_log2+1)
        })

        # https://qiita.com/syoyo/items/ddff3268b4dfa3ebb3d6
        # conv2dの出力サイズw,hが同じ大きさになるようにするには  padding=kernel_size//2 padding_mode='replicate'にする？
        
        self.alpha=alpha
        self.delta_alpha=delta_alpha
        self.upscale= nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,latent_vector: torch.Tensor):
            
        # latent_vectorをnormalizeする
        latent_vector=self.normalize_latent_vector(latent_vector)

        out=self.first_block(latent_vector)

        prev_out=out
        last_res_log2=3
        for res_log2 in range(3,self.current_resolution_log2+1):

            block_module=self.blocks[f"block_{res_log2}"]
            out=block_module(out)

            # 最後の1つ前
            if res_log2== self.current_resolution_log2-1:
                prev_out=out

            last_res_log2=res_log2


        to_rgb=self.to_rgb_layers[f"to_rgb_{last_res_log2}"]
        img=to_rgb(out)

        #   >=3は 現在の出力が4x4じゃないかの確認。4x4の時はto_rgbは1つのはず
        if self.alpha<1 and self.current_resolution_log2 >= 3:
            prev_img=self.to_rgb_layers[f"to_rgb_{last_res_log2-1}"](prev_out)

            upscaled_prev_img= self.upscale(prev_img)

            # ブレンドする
            img = (1-self.alpha)*upscaled_prev_img + self.alpha * img

        return img
        

    def make_block(self,in_channels:int,out_channels:int):
        return nn.Sequential(*[
            nn.Upsample(scale_factor=2, mode='nearest'),
            EqualizedConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            self.norm(),

            EqualizedConv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            self.norm(),
        ])

    def norm(self):
        return PixelNorm()

    def grow(self):
        assert self.current_resolution*2 <= self.resolution ,"これ以上大きくはできない。resolutionを確認"

        # 画像サイズを大きくする
        self.current_resolution=self.current_resolution*2
        self.current_resolution_log2=int(np.log2(self.current_resolution))

        self.alpha=0.0

    def update_alpha(self,delta_alpha=None):
        # alpha更新
        # delta_alpha=Noneの時は initの時に設定した値を用いる
        if delta_alpha is None:
            self.alpha+=self.delta_alpha
        else:
            self.alpha+=self.delta_alpha
        if self.alpha > 1:
            self.alpha=1


class Discriminator(nn.Module):
    def __init__(self,resolution=1024,start_resolution=4,label_size=0,alpha=0.0,delta_alpha=0.001,use_wscale=True) -> None:
        super().__init__()

        self.resolution=resolution
        self.label_size=label_size


        resolution_log2=int(np.log2(resolution))
        assert 2**resolution_log2==resolution, "resolutionは2^xの値を設定する必要あり(e.g. 2^10=1024)"
        self.resolution_log2=resolution_log2

        assert 2**int(np.log2(start_resolution))==start_resolution
        self.current_resolution=start_resolution
        self.current_resolution_log2=int(np.log2(start_resolution))

        self.use_wscale=use_wscale

        activation=nn.LeakyReLU(0.2)

        # [2,resolution_log2]
        self.from_rgb_layers=nn.ModuleDict({
            f"from_rgb_{res_log2}": nn.Sequential(
                EqualizedConv2d(3,num_features(res_log2-1),kernel_size=(1,1)),
                activation,
            ) for res_log2 in range(resolution_log2,1,-1) # こちらは　resolution_log2...2まで
        })

        # resolution_log2 = 5の時　( 32x32　)　のとき
        # [3,resolution_log2]
        self.blocks=nn.ModuleDict({
            f"block_{res_log2}": self.make_block(num_features(res_log2-1),num_features(res_log2-2))  for res_log2 in range(resolution_log2,2,-1)
        })

        self.last_block=nn.Sequential(
            MinibatchStd(), 
            # minibatch stdのoutputのchannel数+1　になるので 次の層のchannels数を+1する
            EqualizedConv2d(num_features(2)+1,num_features(1),kernel_size=(3,3),padding=1),
            activation,
            # (batch_size, 4*4*512)に変形。サイズは変わる？
            nn.Flatten(),

            # (4,4)のサイズのfeatureになっているはず
            EqualizedLinear(4*4*num_features(1),num_features(0)),
            activation,
            EqualizedLinear(num_features(0),1+label_size,gain=1),
        )

        self.downscale2d=  nn.AvgPool2d((2,2))

        self.alpha=alpha
        #  ミニバッチあたらりのalpha増加量。   len(dataset)　を設定で 1epochでalphaが1になり、完全に新しく挿入した層を使うようになる
        self.delta_alpha=delta_alpha 

        
    def forward(self,x):

        input_img=x
        # img => (batch_size, 1 + label_size)　 1は画像が本物か生成されたものか判定。label_sizeはクラスを判別する？

        from_rgb=self.from_rgb_layers[f"from_rgb_{self.current_resolution_log2}"]
        out=from_rgb(x)

        for res_log2 in range(self.current_resolution_log2,2,-1):
            
            block_module=self.blocks[f"block_{res_log2}"]
            # in_size=out.size()
            out=block_module(out)

            # alphaが1でない　かつ 1層目の時
            if  self.alpha < 1 and res_log2 == self.current_resolution_log2:

                downscaled_img=self.downscale2d(input_img)
                out_pre_layer=self.from_rgb_layers[f"from_rgb_{self.current_resolution_log2-1}"](downscaled_img)

                # 前の層の結果と新しく挿入した層の結果をブレンドする
                out=(1-self.alpha)*out_pre_layer + self.alpha * out

        out=self.last_block(out)

        return out
    
    def make_block(self,in_channels:int,out_channels:int):
        return nn.Sequential(*[
            EqualizedConv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            EqualizedConv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),padding=1),
            nn.LeakyReLU(0.2,inplace=True),

            # Downsample
            nn.AvgPool2d((2,2))
        ])
    
    def grow(self):
        assert self.current_resolution*2 <= self.resolution ,f"これ以上大きくはできない。resolutionを確認"

        # 画像サイズを大きくする
        self.current_resolution=self.current_resolution*2
        self.current_resolution_log2=int(np.log2(self.current_resolution))

        # alphaを値を0にして、徐々に1に近づけていく。
        # 最初は前の層の結果から、徐々に新しい層の結果に影響が大きくなるように割合を変えていく
        self.alpha=0

    def update_alpha(self,delta_alpha=None):
        # alpha更新
        # delta_alpha=Noneの時は initの時に設定した値を用いる
        if delta_alpha is None:
            self.alpha+=self.delta_alpha
        else:
            self.alpha+=self.delta_alpha
        if self.alpha > 1:
            self.alpha=1



# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
def count_parameters(model:torch.nn.Module):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        print(name, params)
        total_params+=params
    print(f"Total Trainable Params: {total_params}")
    return total_params

if __name__=="__main__":
    # g=Generator(resolution=1024,start_resolution=4)
    # print(g)
    # x=torch.randn(2,512)
    # out=g(x)
    # print("out",out.size())


    # d=Discriminator(resolution=1024,start_resolution=4)
    # print("D")
    # print(d)
    # x=torch.randn(2,3,4,4)
    # out=d(x)
    # print("out",out.size())
    pass
    # for res_log2 in range(3,5):
    #     _res=2**res_log2

    #     print("res:",_res)
    #     g=Generator(resolution=_res,start_resolution=_res)
    #     g.alpha=1
    #     print(g)
    #     out=torch.randn(3,512)
    #     out=g(out)
    #     print("out size",out.size())
    #     print()

    # resolution=256
    # g=Generator(resolution=resolution,start_resolution=resolution)
    # print("G")
    # count_parameters(g)
    # d=Discriminator(resolution=resolution,start_resolution=resolution)
    # print("D")
    # count_parameters(d)



    # resolution=64
    # start_resolution=4
    # g=Generator(resolution=resolution,start_resolution=start_resolution)
    # print(g)

    # for i in range(6):
    #     print("i:",i)
    #     print("current res",g.current_resolution)
    #     input=torch.randn(1,512)
    #     print(input.size())
        
    #     out=g(input)
    #     print(out.size())

    #     g.grow()



    # resolution=64
    # d=Discriminator(resolution=resolution)
    # print(d)
    # for res_log2 in range(2,6):
    #     print("=current resolution",d.current_resolution)
    #     print("current resolution log2",d.current_resolution_log2)
    #     res=2**res_log2 # 4,8,16,32 ...
    #     img=torch.randn(1,3,res,res)
    #     print(img.size())
    #     out=d(img)
    #     print(out.size())
    #     d.grow()



    # dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
    # input_names = [ "input1" ]
    # output_names = [ "output1" ]
    # torch.onnx.export(d, img, "test_d.onnx", verbose=True, input_names=input_names, output_names=output_names)
