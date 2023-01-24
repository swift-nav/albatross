import gpytorch.kernels as k
import torch
from gpytorch.functions import MaternCovariance


def tens(x):
    return torch.tensor([x], dtype=torch.float64)


def matern(points, lengthscale, order=2.5):
    m = k.MaternKernel(order, lengthscale=lengthscale)
    mc = MaternCovariance()
    xs = tens(points).t()
    return mc.apply(xs, xs, tens(lengthscale), tens(order),
                    lambda x1, x2: m.covar_dist(x1, x2))


LENGTHSCALE=22.2

POINTS = [-100, -10, -5, -2, -1, -1e-2, -1e-5, 0, 1e-5, 1e-2, 1, 2, 5, 10, 100]

if __name__ == "__main__":
    torch.set_printoptions(precision=16)
    print(f"Length scale: {tens(LENGTHSCALE)}")
    print(f"Evaluation points ({len(POINTS)}):")
    print(POINTS)
    print("Matern 5/2:")
    print(matern(POINTS, LENGTHSCALE, order=2.5))
    print("Matern 3/2:")
    print(matern(POINTS, LENGTHSCALE, order=1.5))

