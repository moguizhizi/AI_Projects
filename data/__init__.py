from data.preprocess import StepActionsPreprocessor, StepActionsEnPreprocessor, AutomotiveUserOpinionsPreprocessor
from data.data_loader import StepActionsDataLoader, StepActionsEnDataLoader, AutomotiveUserOpinionsDataLoader
from data.data_processe import StepActionsDataset, StepActionsEnDataset, AutomotiveUserOpinionsDataset
from data.collate_utils import step_actions_collate_fn, step_actions_en_collate_fn, automotive_user_opinions_collate_fn

data_loader = {
    'StepActions': StepActionsDataLoader,
    'StepActions_En': StepActionsEnDataLoader,
    'AutomotiveUserOpinions': AutomotiveUserOpinionsDataLoader
}

load_dataset = {
    'StepActions': StepActionsDataset,
    'StepActions_En': StepActionsEnDataset,
    'AutomotiveUserOpinions': AutomotiveUserOpinionsDataset,
}

load_preprocessor = {
    'StepActions': StepActionsPreprocessor,
    'StepActions_En': StepActionsEnPreprocessor,
    'AutomotiveUserOpinions': AutomotiveUserOpinionsPreprocessor,
}

load_collate = {
    'StepActions': step_actions_collate_fn,
    'StepActions_En': step_actions_en_collate_fn,
    'AutomotiveUserOpinions': automotive_user_opinions_collate_fn,
}



