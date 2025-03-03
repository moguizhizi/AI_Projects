from data.preprocess import StepActionsPreprocessor
from data.data_loader import StepActionsDataLoader
from data.data_processe import StepActionsDataset
from data.collate_utils import step_actions_collate_fn, step_actions_en_collate_fn

data_loader = {
    'StepActions': StepActionsDataLoader,
    'StepActions_En': StepActionsDataLoader,
}

load_dataset = {
    'StepActions': StepActionsDataset,
    'StepActions_En': StepActionsDataset,
}

load_preprocessor = {
    'StepActions': StepActionsPreprocessor,
    'StepActions_En': StepActionsPreprocessor,
}

load_collate = {
    'StepActions': step_actions_collate_fn,
    'StepActions_En': step_actions_en_collate_fn,
}

