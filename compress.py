import argparse

from src.arithmetic_coding import FloatArithmeticCoding
from src.model import FixedProbabilityModel

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

if __name__ == "__main__":
    args = parser.parse_args()
    model = FixedProbabilityModel()
    coder = FloatArithmeticCoding(model)
    in_file = args.input_file
    if args.decompress:
        dec = coder.decode(f"{in_file}", len("ENCODEME"))
    else:
        coder.encode(in_file)
