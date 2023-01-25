import argparse
import numpy as np
import numpy.linalg as la
import pandas as pd
from models import print_output, MISOCP, MIQP_newton

parser = argparse.ArgumentParser(description="Best Subset Selection")
parser.add_argument("--method", default=None, type=str, help="subset selection method")
parser.add_argument("--bigM", default=[], type=float, help="big-M multiplier constant")
parser.add_argument("--dataset", default=None, type=str, help="datafile address")
parser.add_argument("--TimeLimit", default=3600, type=int, help="Time Limit")
args = parser.parse_args()


def compute_bigM(X, y, bigM):
    beta = la.lstsq(X, y, rcond=None)[0]
    return bigM * la.norm(beta, np.inf)


def main():
    # Load and preprocess data
    df = pd.read_csv(args.dataset)
    data = df.values
    X = data[:, 0:-1]
    y = data[:, -1]
    n_features = X.shape[1]
    verbose = True

    # Criteria
    if args.method == "AIC":
        f = lambda i: np.exp(- 2 * i / n_features)
    elif args.method == "BIC":
        f = lambda i: np.exp(- np.log(n_features) * i / n_features)
    else:
        f = lambda i: n_features - i
    
    print(20 * '*')
    bigM = compute_bigM(X, y, args.bigM) if args.bigM else []
    msg = args.dataset[7:-4] + ',' + args.method + ','

    try:
        m1 = MISOCP(X, y, f, bigM, args.TimeLimit, verbose)
        print_output(m1, msg)
    except:
        print(args.dataset[7:-4], args.method, 'Error in MISOCP formulation')

    try:
        m2 = MIQP_newton(X, y, f, bigM, args.TimeLimit, verbose)
        print_output(m2, msg)
    except:
        print(args.dataset[7:-4], args.method, 'Error in MIQP formulation')

    try:
        m3 = MIQP_newton(X, y, f, bigM, args.TimeLimit, verbose, 1)
        print_output(m3, msg)
    except:
        print(args.dataset[7:-4], args.method, 'Error in MIQP to MISOCP formulation')

if __name__ == '__main__':
    main()
