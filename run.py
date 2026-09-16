"""Project command line. Training implementation follows in the next stage."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="MOOC dropout research project")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preprocess", "train", "benchmark", "report"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", choices=("kdd2015", "xuetangx", "oulad"), required=True)
        command.add_argument("--machine", choices=("laptop", "server"), default="laptop")
        if name in ("train", "benchmark"):
            command.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
            command.add_argument("--seeds", type=int, nargs="+", default=[0])
        if name == "train":
            command.add_argument("--model", required=True)
    args = parser.parse_args()
    parser.exit(2, f"{args.command}: scaffold only; this pipeline is not implemented yet.\n")


if __name__ == "__main__":
    main()
