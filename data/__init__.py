from data.preprocess import StepActionsPreprocessor
from data.data_loader import StepActionsDataLoader
from data.data_processe import StepActionsDataset
from data.collate_utils import step_actions_collate_fn

data_loader = {
    'StepActions': StepActionsDataLoader,
}

load_dataset = {
    'StepActions': StepActionsDataset,
}

load_preprocessor = {
    'StepActions': StepActionsPreprocessor,
}

load_collate = {
    'StepActions': step_actions_collate_fn
}

