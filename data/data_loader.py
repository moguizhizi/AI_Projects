from torch.utils.data import DataLoader


class StepActionsDataLoader(DataLoader):
    def __init__(self, dataset, config, collate_fn=None):
        super().__init__(dataset,
                         batch_size=config["batch_size"], shuffle=config["shuffle"], num_workers=config["num_workers"], collate_fn=collate_fn)


class StepActionsEnDataLoader(DataLoader):
    def __init__(self, dataset, config, collate_fn=None):
        super().__init__(dataset,
                         batch_size=config["batch_size"], shuffle=config["shuffle"], num_workers=config["num_workers"], collate_fn=collate_fn)


class AutomotiveUserOpinionsDataLoader(DataLoader):
    def __init__(self, dataset, config, collate_fn=None):
        super().__init__(dataset,
                         batch_size=config["batch_size"], shuffle=config["shuffle"], num_workers=config["num_workers"], collate_fn=collate_fn)
