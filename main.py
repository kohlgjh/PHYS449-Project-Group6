'''Main entry point into code'''
import argparse
from src.model.fusion import FusionNet
from src.visualizer.plotting import plot_train_test


def parse_args():
    '''Parses command line input for arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="integer 1-8 of what feature set to use")
    parser.add_argument("-af", "--activation_function", help="sigmoid or SBAF choice of activation function")
    parser.add_argument('-v', "--verbose", default="True", help="True/False for high/low verbosity")
    return parser.parse_args()

def main(args):
    '''Main code block'''
    verbose = True if args.verbose == "True" else False
    case = int(args.case)
    fusion = FusionNet(args.activation_function, case, verbose)

    train_results = fusion.train()
    test_results = fusion.test()

    # pass results to graphic visualizer (training/testing plot)
    plot_train_test(train_results, test_results)


if __name__ == "__main__":
    args = parse_args()
    main(args)