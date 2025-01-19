from argparse import ArgumentParser

def makeParser():
    """Creates a parser for the program parameters

    Returns:
        parser: an ArgumentParser object containing the program parameters
    """
    parser = ArgumentParser(description="Define the program parameters")

    parser.add_argument('-m', '--mode', type=str, choices=["train", "test"],
                        help="What mode to run the code in")

    parser.add_argument('-d', '--device', type=str, required=False, choices=["cpu", "cuda"], default="cuda",
                        help="What device to use during training and inference. Default is GPU.")
    
    parser.add_argument('-g', '--generator-path', type=str, required=False, default="Generator.pth",
                        help="Where to save the trained generator.")
    
    parser.add_argument('-b', '--batch_size', type=int, required=True,
                        help="What batch size to use while training.")

    return parser