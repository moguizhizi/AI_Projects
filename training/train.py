from utils.file_utils import clear_directory, save_model_weights
from data import load_dataset, load_preprocessor, data_loader, load_collate
from models import load_model
from training import load_loss_func, load_optimizer
from transformers import get_scheduler
import torch
import os
from pprint import pprint


def train(dataset_dict, config_dict, args):
    
    dataset_name = args.dataset
    model_name = args.model_name
    
    # 打印数据集配置
    print(f"Dataset configuration for {dataset_name}:")
    pprint(config_dict[dataset_name])
    
    train_config = config_dict[dataset_name]["train"]

    num_epochs = train_config["epochs"]
    loss_func_type = train_config["loss_func_type"]
    optimizer_type = train_config["optimizer_type"]
    device = config_dict[dataset_name]["device"]
    gradient_accumulation_steps = train_config["gradient_accumulation_steps"]
    use_amp = train_config["use_amp"]
    scheduler_type = train_config["scheduler_type"]
    num_warmup_steps = train_config["num_warmup_steps"]

    preprocessor = load_preprocessor[dataset_name](dataset_dict)
    preprocessor.preprocess()

    dataset = load_dataset[dataset_name](
        dataset_dict[dataset_name]["processed"]["train"], config_dict[dataset_name], model_name)
    collate_fn = load_collate[dataset_name]
    dataloader = data_loader[dataset_name](dataset, train_config, collate_fn)

    model = load_model[model_name](config_dict[dataset_name], args)
    model.to(device)

    loss_func = load_loss_func[loss_func_type](dataset_dict, train_config)
    optimizer = load_optimizer[optimizer_type](
        dataset_dict, train_config, model.parameters())

    # 计算总训练步数（考虑梯度累积）
    steps_per_epoch = len(dataloader) // gradient_accumulation_steps
    if len(dataloader) % gradient_accumulation_steps != 0:
        steps_per_epoch += 1  # 处理余数
    num_training_steps = num_epochs * steps_per_epoch
    # 创建学习率调度器
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    save_dir = os.path.join(
        "AI_Projects", "checkpoints", "model_checkpoints", dataset_name, model_name, "normal")
    clear_directory(save_dir)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(dataloader):
            loss = train_step(model, batch, loss_func,
                              gradient_accumulation_steps, use_amp)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # 清空梯度
                scheduler.step()

            total_loss += loss.item() * gradient_accumulation_steps  # 还原实际 loss

        if len(dataloader) % gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()  # 清空梯度
            scheduler.step()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 调用保存函数
        save_model_weights(model, optimizer, epoch + 1, avg_loss, save_dir=save_dir,
                           file_prefix=f'epoch_{epoch + 1}', save_type='checkpoint')


def train_step(model, batch, loss_func, gradient_accumulation_steps=1, use_amp=False):
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        outputs, labels, task_configs = model(batch)
        loss = loss_func(outputs, labels, task_configs) / \
            gradient_accumulation_steps  # 适配梯度累积

    loss.backward()
    return loss
