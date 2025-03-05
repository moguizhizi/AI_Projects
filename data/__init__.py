from data.preprocess import StepActionsPreprocessor, StepActionsEnPreprocessor
from data.data_loader import StepActionsDataLoader, StepActionsEnDataLoader
from data.data_processe import StepActionsDataset, StepActionsEnDataset
from data.collate_utils import step_actions_collate_fn, step_actions_en_collate_fn

data_loader = {
    'StepActions': StepActionsDataLoader,
    'StepActions_En': StepActionsEnDataLoader,
}

load_dataset = {
    'StepActions': StepActionsDataset,
    'StepActions_En': StepActionsEnDataset,
}

load_preprocessor = {
    'StepActions': StepActionsPreprocessor,
    'StepActions_En': StepActionsEnPreprocessor,
}

load_collate = {
    'StepActions': step_actions_collate_fn,
    'StepActions_En': step_actions_en_collate_fn,
}

