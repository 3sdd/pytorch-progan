import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os
import random
from datetime import datetime,timezone,timedelta
import numpy as np
import copy

import yaml
from tqdm import tqdm

from criterion.loss import WganGeneratorLoss,WganGpDiscriminatorLoss
from models.progan import Generator,Discriminator



# 最後のバッチは捨てている(中途半端な数だとminibatch-std layerでエラーになるので)
def getDataLoader(dataroot:str,image_size:int,batch_size:int,num_workers:int):
    # TODO: 
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = torchvision.datasets.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize あり　なし？
                                transforms.RandomHorizontalFlip(),
                            ]))
    # Create the dataloader
    
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers,pin_memory=False,drop_last=True)

    return dataloader


def disable_debugging_apis():
    # torch.autograd.set_detect_anomaly(mode=False)
    # torch.autograd.profiler.emit_nvtx(enabled=False)
    # torch.autograd.profiler.profile(enabled=False)
    pass

def postprocess_image(fake:torch.Tensor):
    # (batch_size, channel, height, width) で生成された画像のtensor
    
    # -1～1の範囲になるようにする
    out=torch.clamp(fake,min=-1.0, max=1.0)  # normalizeのstd,meanによって変わる


    # TODO: normalizeのstd=0.5,mean=0.5でしているので変更できるようにする 
    # transforms.Normalizeを元に戻している
    out=out*0.5+0.5

    return out

def save_image(net_g_smooth:nn.Module,fixed_noise:torch.Tensor,resolution,writer,grid_nrow=5):
    net_g_smooth.eval()
    with torch.no_grad():
        fake= net_g_smooth(fixed_noise).detach().cpu()
        fake=postprocess_image(fake)
                    
        grid=torchvision.utils.make_grid(fake,padding=2,normalize=True,nrow=grid_nrow)
        torchvision.utils.save_image(grid,f"result/images/generated_resolution{resolution}.png")

        writer.add_image('images/epoch-finished',grid,resolution,dataformats='CHW')

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



# Gs更新するための関数
def update_exponential_moving_average(src_model:Generator,dst_model:Generator,smoothing=0.999):
    with torch.no_grad():
        for src_param,dst_param in zip(src_model.parameters(),dst_model.parameters()):
            p=src_param + (dst_param-src_param)*smoothing    # (1-smoothing)*src_param + smoothing * dst_param
            dst_param.copy_(p)

        # TODO: bufferもコピーする？
        for src_buffer,dst_buffer in zip(src_model.buffers(),dst_model.buffers()):
            b=src_buffer + (dst_buffer-src_buffer)*smoothing
            dst_buffer.copy_(b)



    

def train():
    print("準備中")
    torch.autograd.set_detect_anomaly(True)

    with open('config.yaml','r',encoding='utf-8') as f:
        config=yaml.safe_load(f)
        print(config)

    # randomのseed設定
    seed=config['seed']
    set_seed(seed)

    disable_debugging_apis()

    # gpu高速化の設定
    print("cudnn is available",torch.backends.cudnn.is_available())
    print("cudnn enabled",torch.backends.cudnn.enabled)
    torch.backends.cudnn.benchmark=config['enable_cudnn_benchmark']
    print("cudnn benchmark",torch.backends.cudnn.benchmark)

    os.makedirs('result/',exist_ok=True)
    os.makedirs('result/images',exist_ok=True)
    os.makedirs('result/checkpoints',exist_ok=True)

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:",device)
    latent_size=config['latent_size']
    
    # 学習する枚数
    # これ/データセットの枚数= 学習エポック数
    total_k_images=config['total_k_images']
    phase_training_k_images=config['phase_training_k_images']
    phase_fadein_k_images=config['phase_fadein_k_images']

    label_size=0
    target_resolution= config['target_resolution']
    start_resolution=config['start_resolution']
    dataroot=config['dataroot']
    batch_size_info=config['batch_size_info'] 


    # TODO: 　target_resolution=4 start_resolution=4の時にエラーとなる。直す
    # resolution => bs=? speed   1epochのtime
    # 4 => bs=64   900it/s
    # 8 => bs=64 900it/s  で3mほど
    # 16 => bs=128 300it/s 12m ほど
    # 32 => bs=128 300it/s 12mほど 
    # 64 => bs=128 60it/s 　1h ほど
    # 128 => bs=64 でぎりぎり乗る？　けど厳しそう   1epoch 1h40m
    # 256 =>  bs=32 でいけた ぎりぎりメモリーにのる　12g 。厳しそう　3hほど　20it/s ?

    # 512 => bs= 16     11it/s 1gほど余裕？ 12g占有 5h30m ほど　10it/s
    # 1024 =>  ダメ  bs =  1 でもout of memoryになる 
    # 元コードでのbatch_size  =>  4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4

    # lr=0.0015 # ,0.001
    lr=config['lr']
    # num_workers=8
    # num_workers=os.cpu_count()
    num_workers=4

    # 5x5のグリッドの画像を生成したいので5*5
    fixed_noise = torch.randn(5*5, latent_size, device=device)

    net_g=Generator(resolution=target_resolution,start_resolution=start_resolution)
    net_g_smooth=copy.deepcopy(net_g)
    net_g_smooth.eval()
    net_d=Discriminator(resolution=target_resolution,label_size=label_size,start_resolution=start_resolution)
    current_resolution=start_resolution

    net_g=net_g.to(device)
    net_g_smooth=net_g_smooth.to(device)
    net_d=net_d.to(device)
    print("net generator")
    print(net_g)
    print("net discriminator")
    print(net_d)

    net_g.train()
    net_d.train()

    print("current res:",net_d.current_resolution)
    # バッチサイズ設定
    batch_size=batch_size_info[net_d.current_resolution]
    

    optimizer_g=optim.Adam(net_g.parameters(),lr=lr,betas=(0.0,0.99),eps=1e-8)
    optimizer_d=optim.Adam(net_d.parameters(),lr=lr,betas=(0.0,0.99),eps=1e-8)



    scaler=torch.cuda.amp.GradScaler()
    # Loss
    # g_criterion= nn.BCEWithLogitsLoss()
    g_criterion= WganGeneratorLoss()
    # d_criterion= nn.BCEWithLogitsLoss() 
    d_criterion= WganGpDiscriminatorLoss(discriminator=net_d,gradscaler=scaler) 




    # TODO: dataset celebahq 
    print("データ準備")

    dataloader=getDataLoader(dataroot,net_d.current_resolution,batch_size,num_workers)
    print("データセットサイズ:",len(dataloader.dataset))

    # レイヤー挿入後のalphaの値の変化値 1epoch
    net_d.delta_alpha= batch_size/(phase_fadein_k_images*1000)
    net_g.delta_alpha= batch_size/(phase_fadein_k_images*1000)
    if net_g.current_resolution==4:
        # 4x4の時は前のレイヤーがないので、そのまま学習する
        net_d.alpha=1
        net_g.alpha=1

    # tensorboard
    jst = timezone(timedelta(hours=+9))
    writer=SummaryWriter(log_dir=f"./result/runs/{datetime.now(jst).isoformat().replace(':','_',3)}")

    print("pggan学習開始")
    num_imgs_prev_saved=0
    # 学習に使った枚数
    current_num_imgs=0
    phase_num_imgs=0
    pbar=tqdm(total=(phase_fadein_k_images+phase_training_k_images)*1000,position=0,leave=True)
    # pbar.set_description(f"[Epoch {epoch}/{num_epochs}][resolution={net_g.current_resolution}]")
    pbar.set_description(f"[resolution={net_d.current_resolution}]")

    # previous_resolution=net_d.current_resolution
    start_time=datetime.now(jst)
    # writer.add_text("time/start",f"[epoch {epoch}/{num_epochs}] start time: {start_time.isoformat()}",epoch)
    
    iter_dataloader=iter(dataloader)
    i=1
    while current_num_imgs < total_k_images*1000:

        try:
            img,labels=next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            img,labels=next(iter_dataloader)
        
        # img,labels=next(iter_dataloader)

        # alpha更新
        net_d.update_alpha()
        net_g.update_alpha()
        # fadeinフェーズの時は、前のresolutionの画像と現在のresolutionの画像をブレンドする
        if net_g.alpha<0:
            # 前のresolutionのサイズへリサイズ
            prev_resolution_img=torch.nn.functional.avg_pool2d(img,2)
            # 現在のresolutionのサイズへ戻す
            prev_resolution_img=torch.nn.functional.interpolate(
                img,scale_factor=2,mode="nearest"
            )
            # ブレンドする
            img= net_g.alpha * real_img + (1-net_g.alpha)*prev_resolution_img



        # バッチサイズ
        bs=img.size(0)

        img=img.to(device)
        # labels=labels.to(device)



        real_img=img
        # 
        # Discriminatorの学習 
        # 

        # TODO: discriminatorの学習はgeneratorに比べて多くする？ようなことどこかの論文に書いてあった気がdcgan?
        net_d.zero_grad()
        
        
        with torch.autocast(device_type=device.type):
            # 本物で学習
            output_real=net_d(real_img)


            # TODO: class_scoreも考慮する　　　本物か偽物か+class_label

            # 偽物で学習
            noise=torch.randn(bs,latent_size,device=device)
            # generatorで画像を生成
            fake_img=net_g(noise)

            # ??detach必要？ g側のgradientはここでは計算しないということ？
            output_fake=net_d(fake_img.detach())

            # TODO: unscale?
            err_d=d_criterion(real_img,fake_img,output_real,output_fake)


        # err_d.backward()
        scaler.scale(err_d).backward()
        
        # optimizer_d.step()
        scaler.step(optimizer_d)

        
        # TODO: autocast関連はどうなるの？　特になにもしなくていい？ scaler.unscale_など
        net_g_smooth.alpha=net_g.alpha
        update_exponential_moving_average(net_g,net_g_smooth,smoothing=0.999)

        # 
        #  Generatorの学習
        # 
        
        net_g.zero_grad()
        # label = torch.full((bs,1), real_label, dtype=torch.float, device=device)
        # discriminator学習時に生成した画像を使う


        with torch.autocast(device_type=device.type):
            output=net_d(fake_img)

            # generatorのloss計算
            err_g=g_criterion(output)

        # gradient計算
        # err_g.backward()
        scaler.scale(err_g).backward()
        # generatorを更新
        # optimizer_g.step()
        scaler.step(optimizer_g)


        scaler.update()

        # 学習した枚数更新
        current_num_imgs+=bs
        phase_num_imgs+=bs

        # Output training stats
        # TODO: 調節する
        # if current_num_imgs % batch_size*100 == 0:
        if i%50==0:

            # TODO: lossの + , -
            pbar.set_postfix(alpha_g='{: .5f}'.format(net_g.alpha),alpha_d='{: .5f}'.format(net_d.alpha),loss_g='{: .5f}'.format(err_g.item()),loss_d='{: .5f}'.format(err_d.item()))  # 少数点5位まで表示

            writer.add_scalar('Alpha/G',net_g.alpha,current_num_imgs)
            writer.add_scalar('Alpha/D',net_d.alpha,current_num_imgs)
            writer.add_scalar('Loss/G',err_g.item(),current_num_imgs)
            writer.add_scalar('Loss/D',err_d.item(),current_num_imgs)
            writer.add_scalar('Loss/D_gp',d_criterion.gradient_penalty.item(),current_num_imgs)


        # 約5000枚ごとに画像保存
        if ( (current_num_imgs - num_imgs_prev_saved) > 5000 ):
            net_g_smooth.eval()
            with torch.no_grad():
                fake = net_g_smooth(fixed_noise).detach().cpu()

                #  -1,1の幅にする
                #  0-1の幅に戻す   x/2 + 0.5
                fake=postprocess_image(fake)
                grid=torchvision.utils.make_grid(fake,padding=2,normalize=True,nrow=5)
                torchvision.utils.save_image(grid,f"result/images/generated_iter_{current_num_imgs}.png")

                writer.add_image('images',grid,current_num_imgs,dataformats='CHW')

            num_imgs_prev_saved=current_num_imgs
        


        pbar.update(bs)


        # growするか判定
        # 特定のresolutionで学習が終わった時
        # TODO: growできるか判定
        if (net_d.current_resolution==4 and phase_num_imgs > phase_training_k_images*1000 ) \
            or phase_num_imgs > (phase_training_k_images + phase_fadein_k_images ) *1000:

            # 1phaseにかかった時間をlogに保存しておく
            end_time=datetime.now(jst)
            writer.add_text("time/end",f"[resolution={net_d.current_resolution}] end time: {end_time.isoformat()}",net_d.current_resolution)
            writer.add_text("time/elapsed",f"[resolution={net_d.current_resolution}] elapsed: {(end_time-start_time)}",net_d.current_resolution)

            save_image(net_g_smooth,fixed_noise,net_g.current_resolution,writer,grid_nrow=5)
            # 重みを保存
            # チェックポイント保存
            torch.save(net_g.state_dict(),f"result/checkpoints/checkpoint_targetres{net_g.resolution}_resolution{net_g.current_resolution}.pt")
            torch.save(net_g_smooth.state_dict(),f"result/checkpoints/checkpoint_Gs_targetres{net_g.resolution}_resolution{net_g.current_resolution}.pt")
            # チェックポイント保存
            torch.save({
                "alpha":net_g.alpha, # TODO: paramtgerrに含んで自動的に読み込むようにした
                "resuolution":net_g.resolution, # TODO: parameterに含みたい
                "current_resolution":net_g.current_resolution, # TODO: parameterに含みたい
                "net_g_state_dict":net_g.state_dict(),
                "net_g_smooth_state_dict":net_g_smooth.state_dict(),
                "net_d_state_dict":net_d.state_dict(),
                "optimizer_g_state_dict":optimizer_g.state_dict(),
                "optimizer_d_state_dict":optimizer_d.state_dict(),
                "current_num_imgs":current_num_imgs,
                "phase_num_imgs":phase_num_imgs,
                "total_k_images":total_k_images,
                "fixed_noise":fixed_noise,
            },f"result/checkpoints/checkpoint.pt")

            # 層を増やす
            print(f"[grow]")
            net_d.grow()
            net_g.grow()
            net_g_smooth.grow()

            print(f"resolution:{net_d.current_resolution}")
            # datasetのサイズも変わるように。　transformのresizeだけ変えられないか？
            batch_size=batch_size_info[net_d.current_resolution]
            print("batch size:",batch_size)

            # batchサイズ変わるので新しくdataloader作成
            dataloader=getDataLoader(dataroot,net_d.current_resolution,batch_size,num_workers)
            iter_dataloader = iter(dataloader)

            # batch size変わるのでalphaの更新値も変更する
            net_d.delta_alpha= batch_size/(phase_fadein_k_images*1000)
            net_g.delta_alpha= batch_size/(phase_fadein_k_images*1000)
            net_g_smooth.delta_alpha= batch_size/(phase_fadein_k_images*1000)


            # 新しくlayerを挿入したときにoptimizerの状態を初期化する
            optimizer_g=optim.Adam(net_g.parameters(),lr=lr,betas=(0.0,0.99),eps=1e-8)
            optimizer_d=optim.Adam(net_d.parameters(),lr=lr,betas=(0.0,0.99),eps=1e-8)

            # フェーズの画像枚数をリセットする
            phase_num_imgs=0
            pbar=tqdm(total=(phase_fadein_k_images+phase_training_k_images)*1000,position=0,leave=True)
            pbar.set_description(f"[resolution={net_d.current_resolution}]")

            # スタート時間を記録
            start_time=datetime.now(jst)

        i+=1

    print("学習終了")


if __name__=="__main__":
    train()