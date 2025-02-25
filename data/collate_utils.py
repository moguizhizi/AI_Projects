def step_actions_collate_fn(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    return batch