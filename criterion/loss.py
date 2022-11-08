import torch
import torch.nn.functional as F
import torch.nn as nn

# 参考: https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
#Critic Loss = [average critic score on real images] – [average critic score on fake images]
# Generator Loss = -[average critic score on fake images]
# https://nn.labml.ai/gan/wasserstein/index.html

class WganDiscriminatorLoss(nn.Module):

    # TODO: 1- 部分と1+部分はなぜいるのか？
    # f.reluで値の範囲を制限
    def forward(self,d_output_real:torch.Tensor,d_output_fake:torch.Tensor):
        # returns tuple
        return  F.relu(1-d_output_real).mean() , F.relu(1+d_output_fake).mean()


class WganGpDiscriminatorLoss(nn.Module):
    def __init__(self,discriminator:nn.Module,lambda_gp=10.0,target_gp=1.0,gradscaler=None) -> None:
        super().__init__()
        self.discriminator=discriminator

        self.lambda_gp=lambda_gp
        self.target_gp=target_gp

        self.gradient_penalty=0.0

        self.gradscaler=gradscaler

    def forward(self,real_images:torch.Tensor,fake_images:torch.Tensor,d_output_real:torch.Tensor,d_output_fake:torch.Tensor):
        loss = torch.mean(d_output_fake) -  torch.mean(d_output_real)

        gradient_penalty=self.compute_gradient_penalty(real_images,fake_images.detach())
        gradient_penalty=  gradient_penalty * (self.lambda_gp / self.target_gp**2)
        # gradient penalty保存
        self.gradient_penalty=gradient_penalty

        loss=loss+gradient_penalty
        return loss

    def compute_gradient_penalty(self,real:torch.Tensor,fake:torch.Tensor):
        # real : 本物画像
        # fake: 生成画像
        batch_size=real.size(0)


        with torch.autocast(device_type=real.device.type,enabled=False):
            alpha=torch.rand(batch_size,1,1,1,device=real.device).expand_as(real) 
            alpha.requires_grad_(True)

            interpolated = (1-alpha) * real  + alpha*fake

            out=self.discriminator(interpolated)

            grad_outputs=torch.ones(out.size(),device=real.device)
            # only inputs ?
            gradients=torch.autograd.grad(outputs=out,inputs=interpolated,grad_outputs=grad_outputs,create_graph=True,retain_graph=True, only_inputs=True)[0] 

        if self.gradscaler != None:
            inv_scale = 1.0/self.gradscaler.get_scale()
            gradients=gradients*inv_scale


        gradients=gradients.view(batch_size,-1)
        # print(gradients)
        
        gradients_norm=torch.sqrt(torch.sum(gradients**2,dim=1)+1e-12) # gradients.norm(2,dim=1)
        gradient_penalty=torch.mean((gradients_norm-1.)**2)

        return gradient_penalty

class WganGeneratorLoss(nn.Module):
    # d_output_fakeは discriminator(fake_image)  の結果
    def forward(self,d_output_fake:torch.Tensor):
        # TODO: class_labelを含むときはどうする？ このままでいいのか？
        # TODO: 公式コードでは　acganのlossも使っていた
        # if D.output_shapes[1][1] > 0:
        #     with tf.name_scope('LabelPenalty'):
        #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        #     loss += label_penalty_fakes * cond_weight

        loss= -d_output_fake.mean()

        return loss

