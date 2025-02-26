from utils.file_utils import save_model_weights
from data import load_dataset, load_preprocessor, data_loader, load_collate
from models import load_model
from training import load_loss_func, load_optimizer
import torch


def train(dataset_dict, config_dict, args):
    dataset_name = args.dataset
    model_name = args.model_name

    train_config = config_dict[dataset_name]["train"]

    num_epochs = train_config["epochs"]
    loss_func_type = train_config["loss_func_type"]
    optimizer_type = train_config["optimizer_type"]
    device = config_dict[dataset_name]["device"]
    gradient_accumulation_steps = train_config["gradient_accumulation_steps"]
    use_amp = train_config["use_amp"]

    preprocessor = load_preprocessor[dataset_name](dataset_dict)
    preprocessor.preprocess()

    dataset = load_dataset[dataset_name](
        dataset_dict[dataset_name]["processed"]["train"], model_name)
    collate_fn = load_collate[dataset_name]
    dataloader = data_loader[dataset_name](dataset, train_config, collate_fn)

    model = load_model[model_name](config_dict, args)
    model.to(device)

    loss_func = load_loss_func[loss_func_type](dataset_dict, train_config)
    optimizer = load_optimizer[optimizer_type](
        dataset_dict, train_config, model.parameters())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(dataloader):
            loss = train_step(model, batch, optimizer, loss_func,
                              gradient_accumulation_steps, use_amp)

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # 清空梯度

            total_loss += loss.item() * gradient_accumulation_steps  # 还原实际 loss

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 调用保存函数
        save_model_weights(model, optimizer, epoch + 1, avg_loss, save_dir=f'AI_Projects/checkpoints/model_checkpoints/{dataset_name}/{model_name}',
                           file_prefix=f'epoch_{epoch + 1}', save_type='checkpoint')


def train_step(model, batch, optimizer, loss_func, gradient_accumulation_steps=1, use_amp=False):
    """执行一个训练步骤，支持梯度累积和混合精度训练"""
    optimizer.zero_grad()

    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        outputs, labels = model(batch)
        loss = loss_func(outputs, labels) / \
            gradient_accumulation_steps  # 适配梯度累积

    loss.backward()
    return loss
