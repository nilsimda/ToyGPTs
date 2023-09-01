import argparse

from src.arithmetic_coding import ArithmeticCoding
from src.model import GPT2, FixedProbabilityModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog="CompressGPT",
        description="Compression and decompression of a text file with GPT2",
    )
    parser.add_argument(
        "input_file",
        metavar="input_file",
        type=str,
        help="The filename of the text file to (de)compress",
    )

    parser.add_argument(
        "-d",
        "--decompress",
        action="store_true",
        help="Decompress the file instead of compressing it",
    )
    args = parser.parse_args()
    return args.input_file, args.decompress


def main():
    # input_file, decompress_flag = parse_arguments()
    model = GPT2()
    context_ids = model.tokenizer(
        "how many moons does earth have? ", return_tensors="pt"
    )
    print(context_ids)
    next_token_id = model.tokenizer("one", return_tensors="pt")["input_ids"][0, 0]
    print(next_token_id)
    prob = model.getProbas(context_ids, next_token_id)
    print(prob)


if __name__ == "__main__":
    main()
    """
    coder = (model)
    in_file = args.input_file
    if args.decompress:
        dec = coder.decode(f"{in_file}", len("ENCODEME"))
    else:
        coder.encode(in_file)
    """
