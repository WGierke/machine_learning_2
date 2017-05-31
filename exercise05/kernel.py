import argparse
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--length', help='Length of time series', required=True)
    parser.add_argument('-n', '--amount', help='Amount of time series', required=True)
    parser.add_argument('-i', '--iterations', help='Iterations that should be performed', required=True)
    args = parser.parse_args()

    n, d = int(args.length), int(args.amount)

    iterations = int(args.iterations)
    for _ in tqdm(range(iterations)):
        X = np.random.rand(d, n)
        c = np.random.rand(n)

        sum_ = 0
        for i in range(n):
            for j in range(n):
                convolution = np.convolve(X[i, :], X[j, :], 'same')
                squared_norm = np.square(np.linalg.norm(convolution))
                squared_norm *= c[i] * c[j]
                sum_ += squared_norm
        if sum_ < 0:
            print("Kernel value is non-positive")
            return

    print("Could not find non-positive kernel value")


if __name__ == '__main__':
    main()
