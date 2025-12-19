import pdb
import torch
import time
import os
import sys
from util.utils import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from Conv_TasNet import check_parameters
from graph import create_graph
from PyG import graph_to_pyg_data
from dataLoader.imageReader import ImageReader, read_img
import csv
from imageloss import ImageSeparationLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts



def denormalize(tensor, mean=(0.402, 0.415, 0.386), std=(0.117, 0.116, 0.112)):
    """
    反归一化图像张量，用于还原已经用ImageNet统计数据归一化的图像
    """
    if not isinstance(tensor, torch.Tensor):
        return tensor

    mean = torch.tensor(mean, device=tensor.device)
    std = torch.tensor(std, device=tensor.device)

    # Handle different tensor dimensions
    if tensor.dim() == 4:  # batched images
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif tensor.dim() == 3:  # single image
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)

    return tensor * std + mean


# def color_correction(pred, ref):
#     """对预测图像进行颜色校正，使其更接近参考图像的颜色分布"""
#     # 计算参考图像的均值和标准差 (按通道)
#     ref_mean = torch.mean(ref, dim=(2, 3), keepdim=True)
#     ref_std = torch.std(ref, dim=(2, 3), keepdim=True)
#
#     # 计算预测图像的均值和标准差
#     pred_mean = torch.mean(pred, dim=(2, 3), keepdim=True)
#     pred_std = torch.std(pred, dim=(2, 3), keepdim=True)
#
#     # 应用颜色校正
#     corrected = (pred - pred_mean) * (ref_std / (pred_std + 1e-6)) + ref_mean
#     return torch.clamp(corrected, 0, 1)
def color_correction(preds, refs):
    """支持单图或批量图像的颜色校正"""
    if isinstance(preds, list):
        preds = torch.stack(preds, dim=0)
    if isinstance(refs, list):
        refs = torch.stack(refs, dim=0)

    if preds.dim() == 3:
        preds = preds.unsqueeze(0)
    if refs.dim() == 3:
        refs = refs.unsqueeze(0)

    ref_mean = torch.mean(refs, dim=(2, 3), keepdim=True)
    ref_std = torch.std(refs, dim=(2, 3), keepdim=True)
    pred_mean = torch.mean(preds, dim=(2, 3), keepdim=True)
    pred_std = torch.std(preds, dim=(2, 3), keepdim=True)

    corrected = (preds - pred_mean) * (ref_std / (pred_std + 1e-6)) + ref_mean
    return torch.clamp(corrected, 0, 1)


def to_device(dicts, device):
    '''
       load dict data to cuda
    '''

    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        elif isinstance(datas, list):
            return [data.to(device) for data in datas]
        else:
            raise RuntimeError('datas is not torch.Tensor and list type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')


class Trainer():
    '''
       Trainer of Conv-Tasnet
       input:
             net: load the Conv-Tasnet model
             checkpoint: save model path
             optimizer: name of opetimizer
             gpu_ids: (int/tuple) id of gpus
             optimizer_kwargs: the kwargs of optimizer
             clip_norm: maximum of clip norm, default: None
             min_lr: minimun of learning rate
             patience: Number of epochs with no improvement after which learning rate will be reduced
             factor: Factor by which the learning rate will be reduced. new_lr = lr * factor
             logging_period: How long to print
             resume: the kwargs of resume, including path of model, Whether to restart
             stop: Stop training cause no improvement
    '''

    def __init__(self,
                 net,
                 checkpoint="checkpoint",
                 optimizer="adamw",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 stop=1000,
                 num_epochs=150,
                 save_by_epoch=False,
                 save_by_train_loss=True,
                 lr_scheduler_config=None,
                 early_stopping_config=None
                 ):
        resume = resume or {'resume_state': False}  # ✅ 修改: 防止 resume 为 None

        # if the cuda is available and if the gpus' type is tuple
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid

        # mkdir the file of Experiment path
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        # build the logger object
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=False)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # current epoch
        self.stop = stop
        self.best_val_loss = float('inf')
        self.early_stopping_config = early_stopping_config or {
            'patience': 20,
            'min_delta': 0.001
        }
        self.stop = self.early_stopping_config['patience']
        self.min_delta = self.early_stopping_config['min_delta']
        self.epochs_without_improvement = 0
        self.best_epoch = 0

        # 初始化对抗损失
        

        # Whether to resume the model
        if resume['resume_state']:
            cpt = torch.load(os.path.join(
                resume['path'], self.checkpoint, 'best.pt'), map_location='cpu')
            self.cur_epoch = cpt['epoch']
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume['path'], self.cur_epoch))
            net.load_state_dict(cpt['model_state_dict'])
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs, state=cpt['optim_state_dict'])
        else:
            self.net = net.to(self.device)
            torch.cuda.empty_cache()
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        # check model parameters
        self.param = check_parameters(self.net)

        # Reduce lr
        # ReduceLROnPlateau scheduler starting from epoch 18
        lr_scheduler_config = lr_scheduler_config or {
            'type': 'plateau',
            'patience': 5,
            'factor': 0.5
        }
    
        if lr_scheduler_config['type'] == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=lr_scheduler_config.get('t0', 10),
                T_mult=lr_scheduler_config.get('t_mult', 2),
                )
        elif lr_scheduler_config['type'] == 'plateau':
            # 使用ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=lr_scheduler_config.get('factor', factor), 
                patience=lr_scheduler_config.get('patience', patience), 
                min_lr=min_lr,
                verbose=True
            )

        # logging
        self.logger.info("Starting preparing model ............")
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            self.gpuid, self.param))
        self.clip_norm = clip_norm
        # clip norm
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

        # number of epoch
        self.num_epochs = num_epochs
        self.mse = torch.nn.MSELoss()

        self.save_by_epoch = save_by_epoch
        self.save_by_train_loss = save_by_train_loss
        self.sample_dir = os.path.join(checkpoint, "samples")
        os.makedirs(self.sample_dir, exist_ok=True)

    def create_optimizer(self, optimizer, kwargs, state=None):
        '''
           create optimizer
           optimizer: (str) name of optimizer
           kwargs: the kwargs of optimizer
           state: the load model optimizer state
        '''
        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adamw": torch.optim.AdamW,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.net.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def save_checkpoint(self, is_best_train=False, is_best_val=False, is_last=False, is_epoch=False, epoch_num=None):
        '''改进的检查点保存，支持更多保存选项'''
        if is_best_train:
            checkpoint_name = "best_train_loss"
        elif is_best_val:
            checkpoint_name = "best_val_loss"
        elif is_last:
            checkpoint_name = "last"
        elif is_epoch and epoch_num is not None:
            checkpoint_name = f"epoch_{epoch_num:03d}"
        else:
            return  # 如果没有指定保存类型，则不保存

        save_path = os.path.join(self.checkpoint, checkpoint_name + ".pt")

        # 保存模型、优化器状态和当前epoch
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss if hasattr(self, 'best_val_loss') else float('inf'),
            },
            save_path
        )

        self.logger.info(f"Saved checkpoint: {checkpoint_name}")

    # def save_sample_images(self, batch, outputs, prefix="train"):
    #     """保存样本图像进行可视化，包含反归一化处理"""
    #     # 选择几个图像进行可视化
    #     sample_idx = 0
    #
    #     # 获取混合和参考图像并反归一化
    #     mixed = (batch['mix'][sample_idx].detach().cpu())
    #     refs = [(ref[sample_idx].detach().cpu()) for ref in batch['ref']]
    #     '''
    #     # 计算每个参考图像的mean和std
    #     ref_means = []
    #     ref_stds = []
    #     for ref in refs:
    #         # 计算参考图像的均值和标准差
    #         ref_mean = torch.mean(ref, dim=(1, 2))
    #         ref_std = torch.std(ref, dim=(1, 2))
    #         ref_means.append(ref_mean)
    #         ref_stds.append(ref_std)
    #
    #     # 获取分离的输出并使用对应的source的mean和std进行反归一化
    #     preds = []
    #     for i, out in enumerate(outputs):
    #         out_tensor = out[sample_idx].detach().cpu()
    #         # 先使用默认的ImageNet均值和标准差进行反归一化
    #         pred_default = denormalize(out_tensor)
    #
    #         # 然后使用对应的source的mean和std进行颜色校正
    #         if i < len(ref_means):
    #             pred_src_specific = denormalize(out_tensor, mean=ref_means[i], std=ref_stds[i])
    #             preds.append(pred_src_specific)
    #     '''
    #     preds = [(out[sample_idx].detach().cpu()) for out in outputs]
    #     # 确保所有值都在有效的图像范围内
    #     mixed = torch.clamp(mixed, 0.0, 1.0)
    #     refs = [torch.clamp(ref, 0.0, 1.0) for ref in refs]
    #     preds = [torch.clamp(pred, 0.0, 1.0) for pred in preds]
    #
    #     # 也保存单独的图像以便更好地检查
    #     if self.cur_epoch % 2 == 0:  # 每2个epoch保存单独图像
    #         indiv_dir = os.path.join(self.sample_dir, f"epoch_{self.cur_epoch:03d}")
    #         os.makedirs(indiv_dir, exist_ok=True)
    #
    #         # 保存混合输入
    #         vutils.save_image(mixed, os.path.join(indiv_dir, f"{prefix}_mixed.png"))
    #
    #         # 保存参考图像
    #         for i, ref in enumerate(refs):
    #             vutils.save_image(ref, os.path.join(indiv_dir, f"{prefix}_ref{i + 1}.png"))
    #
    #         # 保存预测图像
    #         for i, pred in enumerate(preds):
    #             vutils.save_image(pred, os.path.join(indiv_dir, f"{prefix}_pred{i + 1}.png"))
    #
    #             # 保存差异图（可选）
    #             if i < len(refs):
    #                 diff = torch.abs(refs[i] - pred)  # 放大差异
    #                 # diff = torch.clamp(diff, 0.0, 1.0)
    #                 vutils.save_image(diff, os.path.join(indiv_dir, f"{prefix}_diff{i + 1}.png"))
    #
    #         corrected_preds = []
    #         for i, pred in enumerate(preds):
    #             if i < len(refs):
    #                 # corrected = color_correction(pred.unsqueeze(0), refs[i].unsqueeze(0)).squeeze(0)
    #                 refs = torch.stack(refs, dim=0)  # [49, 3, 256, 256]
    #                 corrected = color_correction(pred, refs)
    #
    #                 corrected_preds.append(corrected)
    #
    #         # 同时保存原始和校正后的结果
    #         for i, (pred, corrected) in enumerate(zip(preds, corrected_preds)):
    #             vutils.save_image(pred, os.path.join(indiv_dir, f"{prefix}_pred{i + 1}.png"))
    #             vutils.save_image(corrected, os.path.join(indiv_dir, f"{prefix}_pred{i + 1}_corrected.png"))

    def save_sample_images(self, batch, outputs, prefix="train"):
        """保存样本图像进行可视化，包含反归一化和颜色校正处理"""
        sample_idx = 0  # 选第一个样本展示

        # 获取混合图像和参考图像（refs 为 List[Tensor[C,H,W]]）
        mixed = batch['mix'][sample_idx].detach().cpu()
        refs = [ref[sample_idx].detach().cpu() for ref in batch['ref']]
        preds = list(outputs[sample_idx].detach().cpu())  # outputs[sample_idx] shape: [num_spks, 3, H, W]


        # --- 调试信息 ---
        print(f"[Debug] mixed shape: {mixed.shape}")  # [3, H, W]
        print(f"[Debug] refs[0] shape: {refs[0].shape}  refs length: {len(refs)}")
        print(f"[Debug] preds[0] shape: {preds[0].shape}  preds length: {len(preds)}")

        # 归一化图像（防止溢出）
        mixed = torch.clamp(mixed, 0.0, 1.0)
        refs = [torch.clamp(ref, 0.0, 1.0) for ref in refs]
        preds = [torch.clamp(pred, 0.0, 1.0) for pred in preds]

        # 每2个 epoch 保存一次图像
        if self.cur_epoch % 2 == 0:
            indiv_dir = os.path.join(self.sample_dir, f"epoch_{self.cur_epoch:03d}")
            os.makedirs(indiv_dir, exist_ok=True)

            # 保存混合输入
            vutils.save_image(mixed, os.path.join(indiv_dir, f"{prefix}_mixed.png"))

            # 保存参考图像
            for i, ref in enumerate(refs):
                vutils.save_image(ref, os.path.join(indiv_dir, f"{prefix}_ref{i + 1}.png"))

            # 保存预测图像（原图）
            for i, pred in enumerate(preds):
                vutils.save_image(pred, os.path.join(indiv_dir, f"{prefix}_pred{i + 1}.png"))

            # 保存差异图
            for i in range(min(len(preds), len(refs))):
                diff = torch.abs(preds[i] - refs[i])
                vutils.save_image(diff, os.path.join(indiv_dir, f"{prefix}_diff{i + 1}.png"))

            # --- 批量颜色校正 ---
            try:
                pred_tensor = torch.stack(preds, dim=0)  # [N, C, H, W]
                ref_tensor = torch.stack(refs, dim=0)  # [N, C, H, W]

                print(f"[Debug] pred_tensor shape: {pred_tensor.shape}")
                print(f"[Debug] ref_tensor shape: {ref_tensor.shape}")

                corrected_preds = color_correction(pred_tensor, ref_tensor)  # [N, C, H, W]
            except Exception as e:
                print("[Error] during color correction:", str(e))
                corrected_preds = preds  # fallback: 原图

            # 保存颜色校正后的预测图
            for i, corrected in enumerate(corrected_preds):
                vutils.save_image(corrected, os.path.join(indiv_dir, f"{prefix}_pred{i + 1}_corrected.png"))

    def train(self, train_dataloader):
        '''
           training model
        '''
        self.logger.info('Training model ......')
        losses = []
        d_losses = []  # 记录判别器损失
        start = time.time()
        current_step = 0
        loss_fn = ImageSeparationLoss().to("cuda:0")
        
        #if self.cur_epoch >= 20:
        #    loss_fn.set_fixed_order(True)
        #else:
        #    loss_fn.set_fixed_order(False)
        loss_fn.set_fixed_order(False)
        self.net.train()
       
        

        for egs in train_dataloader:
            current_step += 1
            egs = to_device(egs, self.device)
            self.optimizer.zero_grad()

            # 创建图结构并前向传播
            G_global, G_local = create_graph(egs['mix'], window_size=[64, 64], step_size=[32, 32])
            # G_global, G_local = create_graph(egs['mix'], window_size=[8, 8], step_size=[8, 8])
            data_global = graph_to_pyg_data(G_global)
            data_local = graph_to_pyg_data(G_local)

            # ────────── ① 前向得到网络输出 ──────────
            outs = data_parallel(self.net, (egs['mix'], data_global, data_local), device_ids=self.gpuid)

            # ────────── ② 拆分 outputs 和 masks ──────────
            if isinstance(outs, tuple):  # 模型返回 (separated_imgs, masks)
                ests, masks = outs  # ests: [B, num_spks, 3,H,W]   masks: [B,num_spks,128,H,W]
            else:
                ests, masks = outs, None  # 兼容旧模型

            # ────────── ③ 计算损失时把 masks 传进去 ──────────
            loss = loss_fn(ests, egs, masks=masks)

            loss.backward()

            # ✅ 插入此处，打印每一层的梯度最大值与均值
            print(f"\n[Epoch {self.cur_epoch} | Step {current_step}] Gradient Stats:")
            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    grad_max = param.grad.abs().max().item()
                    grad_mean = param.grad.abs().mean().item()
                    print(f"  {name:<40} | grad max: {grad_max:.4e}, mean: {grad_mean:.4e}")


            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()

            losses.append(loss.item())
            if current_step % 2 == 0:
                self.save_sample_images(egs, ests)
            if len(losses) % self.logging_period == 0:
                avg_loss = sum(
                    losses[-self.logging_period:]) / self.logging_period
                self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses) / len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end - start) / 60))
        return total_loss_avg

    def val(self, val_dataloader):
        '''
           validation model
        '''
        self.logger.info('Validation model ......')
        self.net.eval()
        losses = []
        loss_fn = ImageSeparationLoss().to("cuda:0")
        loss_fn.set_fixed_order(False)
        #if self.cur_epoch >= 20:
           # loss_fn.set_fixed_order(True)
        #else:
           # loss_fn.set_fixed_order(False)
        current_step = 0
        start = time.time()
        with torch.no_grad():
            for egs in val_dataloader:
                current_step += 1
                egs = to_device(egs, self.device)  # 将变量 egs 移动到指定的设备上
                G_global, G_local = create_graph(egs['mix'], window_size=[64, 64], step_size=[32, 32])
                # G_global, G_local = create_graph(egs['mix'], window_size=[8, 8], step_size=[8, 8])

                data_global = graph_to_pyg_data(G_global)
                data_local = graph_to_pyg_data(G_local)
                # ────────── ① 前向得到网络输出 ──────────
                outs = data_parallel(self.net, (egs['mix'], data_global, data_local), device_ids=self.gpuid)

                # ────────── ② 拆分 outputs 和 masks ──────────
                if isinstance(outs, tuple):  # 模型返回 (separated_imgs, masks)
                    ests, masks = outs  # ests: [B, num_spks, 3,H,W]   masks: [B,num_spks,128,H,W]
                else:
                    ests, masks = outs, None  # 兼容旧模型

                # ────────── ③ 计算损失时把 masks 传进去 ──────────
                loss = loss_fn(ests, egs, masks=masks)

                losses.append(loss.item())
                if current_step % 2 == 0:
                    self.save_sample_images(egs, ests, prefix="val")
                if len(losses) % self.logging_period == 0:
                    avg_loss = sum(
                        losses[-self.logging_period:]) / self.logging_period
                    self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                        self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses) / len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end - start) / 60))
        return total_loss_avg

    def run(self, train_dataloader, val_dataloader):
        train_losses = []
        val_losses = []

        with torch.cuda.device(self.gpuid[0]):
            # 初始保存最后一个检查点
            self.save_checkpoint(is_last=True)

            # 初始验证
            val_loss = self.val(val_dataloader)
            train_loss = float('inf')  # 初始训练损失设为无穷大

            # 设置初始最佳损失
            best_val_loss = val_loss
            best_train_loss = float('inf')
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.best_epoch = self.cur_epoch

            self.logger.info("Starting epoch from {:d}, val_loss = {:.4f}".format(
                self.cur_epoch, val_loss))

            # 训练循环
            while self.cur_epoch < self.num_epochs:
                self.cur_epoch += 1

                # 训练一个epoch
                train_loss = self.train(train_dataloader)

                # 验证
                val_loss = self.val(val_dataloader)

                # 记录损失
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # 更新学习率
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                # 检查验证损失改进
                if val_loss < best_val_loss - self.min_delta:  # 使用min_delta
                    self.epochs_without_improvement = 0
                    best_val_loss = val_loss
                    self.best_val_loss = val_loss
                    self.best_epoch = self.cur_epoch
                    self.save_checkpoint(is_best_val=True)
                    self.logger.info('Epoch: {:d}, New best val_loss: {:.4f}'.format(
                        self.cur_epoch, best_val_loss))
                else:
                    self.epochs_without_improvement += 1
                    self.logger.info('No improvement for {:d} epochs, best val_loss: {:.4f}'.format(
                        self.epochs_without_improvement, best_val_loss))

                # 检查训练损失改进
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    if self.save_by_train_loss:
                        self.save_checkpoint(is_best_train=True)
                        self.logger.info('Epoch: {:d}, New best train_loss: {:.4f}'.format(
                            self.cur_epoch, best_train_loss))

                # 每个epoch保存模型（如果启用）
                if self.save_by_epoch:
                    self.save_checkpoint(is_epoch=True, epoch_num=self.cur_epoch)

                # 保存最后的检查点
                self.save_checkpoint(is_last=True)

                # 早停检查
                if self.epochs_without_improvement >= self.stop:
                    self.logger.info("Early stopping triggered! No improvement for {:d} epochs".format(
                        self.epochs_without_improvement))
                    break

            self.logger.info("Training completed for {:d}/{:d} epochs! Best epoch: {:d} with val_loss: {:.4f}".format(
                self.cur_epoch, self.num_epochs, self.best_epoch, best_val_loss))

            # 绘制损失曲线
            plt.figure(figsize=(10, 6))
            plt.title("Loss of train and test")
            x = [i + 1 for i in range(len(train_losses))]  # 从1开始的epoch编号
            plt.plot(x, train_losses, 'b-', label='train_loss', linewidth=1.0)
            plt.plot(x, val_losses, 'c-', label='val_loss', linewidth=1.0)
            plt.legend()
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(os.path.join(self.checkpoint, 'loss_curve.png'), dpi=300)
            plt.close()