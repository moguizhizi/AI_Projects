import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset", type=str, required=True, default="StepActions")
    parser.add_argument("--model_name", type=str, required=True, default='bert-base-chinese')
    
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)

    return parser


def get_args():
    return parse_args().parse_args()
