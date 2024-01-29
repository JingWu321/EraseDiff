import argparse
import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from train_scripts.convertModels import savemodelDiffusers
from train_scripts.dataset import (
    setup_nsfw_data,
    setup_model,
)
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm

import gc
from timm.utils import AverageMeter


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def get_param(net):
    new_param = []
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            new_param.append(param.clone())
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return new_param


def set_param(net, old_param):
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            param.copy_(old_param[j])
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return net



def erasediff(
    unl_eps,
    K_steps,
    lambda_bome,
    rem_num,
    train_method,
    batch_size,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    sampler = DDIMSampler(model)

    criteria = torch.nn.MSELoss()
    forget_dl, remain_dl = setup_nsfw_data(
        batch_size, forget_path='./dataFolder/NSFW',
        remain_path='./dataFolder/NotNSFW', image_size=image_size)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                print(name)
                parameters.append(param)
        # train all layers
        if train_method == "full":
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                print(name)
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    print(name)
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    print(name)
                    parameters.append(param)
    # set model to train
    model.train()

    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-nsfw-noiseN-mask-method_{train_method}-Ksteps_{K_steps}-lambdabome_{lambda_bome}-lr_{lr}"
    else:
        name = f"compvis-nsfw-noiseN-method_{train_method}-Ksteps_{K_steps}-lambdabome_{lambda_bome}-lr_{lr}"

    # NSFW Removal
    word_wear = "a photo of a person wearing clothes"
    word_print = 'nsfw'.replace(" ", "")

    # TRAINING CODE
    step = 0
    for epoch in range(unl_eps):

        model.train()
        with tqdm(total=len(forget_dl)) as time:

            param_i = get_param(model) # get \theta_i
            # [1] K-steps: get \theta_i^K, train over the data via the ambiguous labels, line 1 in Algorithm 1 (BOME!)
            for j in range(K_steps):
                unl_losses = AverageMeter()
                for i, images in enumerate(forget_dl):
                    optimizer.zero_grad()
                    forget_batch = next(iter(forget_dl))

                    forget_input, forget_emb = model.get_input(
                        forget_batch, model.first_stage_key
                    )

                    t = torch.randint(
                        0,
                        model.num_timesteps,
                        (forget_input.shape[0],),
                        device=model.device,
                    ).long()
                    noise = torch.rand_like(forget_input, device=model.device)
                    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                    forget_out = model.apply_model(forget_noisy, t, forget_emb)

                    forget_loss = criteria(forget_out, noise)
                    forget_loss.backward()
                    if mask_path:
                        for n, p in model.named_parameters():
                            if p.grad is not None and n in parameters:
                                p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                    device
                                )
                                print(n)
                    optimizer.step()

                    unl_losses.update(forget_loss)
                    torch.cuda.empty_cache()
                    gc.collect()


                model = set_param(model, param_i) # keep \theta_i for f and q
                # [2] Update \theta_{i+1}
                used_num_samples = 0
                for i, images in enumerate(forget_dl):
                    model.train()
                    optimizer.zero_grad()

                    forget_batch = next(iter(forget_dl))
                    remain_batch = next(iter(remain_dl))

                    loss_dr = model.shared_step(remain_batch)[0]
                    forget_input, forget_emb = model.get_input(
                        forget_batch, model.first_stage_key
                    )

                    t = torch.randint(
                        0,
                        model.num_timesteps,
                        (forget_input.shape[0],),
                        device=model.device,
                    ).long()
                    noise = torch.rand_like(forget_input, device=model.device)

                    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                    forget_out = model.apply_model(forget_noisy, t, forget_emb)
                    loss_du = criteria(forget_out, noise)

                    loss_q = loss_du - unl_losses.avg.detach()  # line 2 in Algorithm 1 (BOME!)
                    # [3] Get lambda_bome
                    # TODO

                    # [4] Update the model parameters # line 3 in Algorithm 1 (BOME!)
                    loss = loss_dr + lambda_bome * loss_q
                    loss.backward()
                    losses.append(loss.item() / batch_size)
                    if mask_path:
                        for n, p in model.named_parameters():
                            if p.grad is not None and n in parameters:
                                p.grad *= mask[n.split("model.diffusion_model.")[-1]].to(
                                    device
                                )
                                print(n)
                    optimizer.step()
                    step += 1
                    if (step+1) % 20 == 0:
                        print(f"step: {i}, unl_loss: {unl_losses.avg.detach():.4f}, loss_du: {loss_du:.4f}, loss_q: {loss_q:.4f}, loss_dr: {loss_dr:.4f}, loss: {loss:.4f}")
                        save_history(losses, name, word_print)
                        model.eval()
                        save_model(model, name, step - 1, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)

                    time.set_description("Epoch %i" % epoch)
                    time.set_postfix(loss=loss.item() / batch_size)
                    sleep(0.1)
                    time.update(1)

                    # used_num_samples += remain_images.shape[0]
                    # if used_num_samples > rem_num:
                    #     break

    model.eval()
    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name, word_print)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print("Saving Model in Diffusers Format")
        savemodelDiffusers(
            name, compvis_config_file, diffusers_config_file, device=device, num=num,
        )

def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TrainEraseDiff",
        description="Finetuning stable diffusion model to erase concepts using EraseDiff method",
    )

    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--K_steps",
        type=int,
        required=False,
        default=2,
    )
    parser.add_argument(
        "--lambda_bome",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--rem_num",
        type=int,
        required=False,
        default=4096,
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0,0",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()

    train_method = args.train_method
    batch_size = args.batch_size
    unl_eps = args.epochs
    K_steps = args.K_steps
    lambda_bome = args.lambda_bome
    rem_num = args.rem_num
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    erasediff(
        train_method=train_method,
        batch_size=batch_size,
        unl_eps=unl_eps,
        K_steps=K_steps,
        lambda_bome=lambda_bome,
        rem_num=rem_num,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask_path=mask_path,
        diffusers_config_path=diffusers_config_path,
        device=device,
        image_size=image_size,
        ddim_steps=ddim_steps,
    )
