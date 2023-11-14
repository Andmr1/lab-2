import numpy as np
from numpy.random import normal
import statistics as st
import pandas as pd
import scipy.stats as sts
import matplotlib.pyplot as plt

M = 3
sg = 2
N = 70
k = 4
gamma = 0.95


def find_statistic_formula(array: np.ndarray) -> None:
    X = 0
    D = 0
    M3 = 0
    M4 = 0
    for i in array:
        X += i
    X /= N
    for i in array:
        D += (i - X) ** 2
    D /= N
    S = (N * D) / (N - 1)
    for i in array:
        M3 += (i - X) ** 3
    M3 /= N
    for i in array:
        M4 += (i - X) ** 4
    M4 /= N
    sigma3 = D ** 1.5
    sigma4 = D ** 2
    A = M3 / sigma3
    E = (M4 / sigma4) - 3
    R = array[N-1] - array[0]
    n = N//2
    Me = (array[n-1] + array[n]) / 2
    ar = np.hsplit(array, 2)
    ar1 = ar[0]
    ar2 = ar[1]
    n = n//2
    IQR = ar2[n+1] - ar1[n+1]
    print("Data found with formulas:\n")
    print("X = ", X, "\n")
    print("D = ", D, "\n")
    print("S = ", S, "\n")
    print("sigma = ", D**0.5, "\n")
    print("A = ", A, "\n")
    print("E = ", E, "\n")
    print("R = ", R, "\n")
    print("Me = ", Me, "\n")
    print("IQR = ", IQR, "\n")


def create_normal_sample(n: int, loc: float, scale: float):
    return np.random.normal(loc=loc, scale=scale, size=n)


def create_student_sample(n: int, k: int):
    return np.random.standard_t(k, n)


def show_cdf(array: np.ndarray) -> None:
    ar = array
    ar.sort()
    x = np.linspace(ar[0] - 2, ar[ar.size - 1] + 2, 100000)
    y = sts.norm.cdf(x, M, sg)
    plt.plot(x, y)
    y2 = sts.ecdf(array)
    ax = plt.subplot()
    y2.cdf.plot(ax)
    plt.show()


def show_hist_and_pdf(array: np.ndarray) -> None:
    ar = array
    ar.sort()
    x = np.linspace(ar[0] - 2, ar[ar.size - 1] + 2, 10000)
    y = sts.norm.pdf(x, M, sg)
    plt.plot(x, y)
    plt.hist(array, bins="fd", density=True, color="red")
    plt.show()


def show_absolute_freq(array: np.ndarray)->None:
    hst, bin_edg = np.histogram(array, bins="fd", density=False)
    lens = np.array([])
    for i in range(bin_edg.size - 1):
        lens = np.append(lens, (bin_edg[i+1] - bin_edg[i]))
    middles = np.array([])
    for i in range(lens.size):
        middles = np.append(middles, bin_edg[i] + lens[i]/2.0)
    Mx = st.mean(array)
    sigma = st.stdev(array)
    theory_func = sts.norm.pdf(middles, Mx, sigma)
    y = np.array([])
    for i in range(hst.size):
        y = np.append(y, (lens[i]*N*theory_func[i]))
    plt.plot(middles, y)
    plt.hist(array, bins="fd", density=False)
    plt.show()


def show_box_plot(array: np.ndarray)->None:
    d = {"Box": array}
    pd.DataFrame(d).boxplot(showmeans=True)
    plt.show()


def show_stats_func(array: np.ndarray) -> None:
    print("Statistics found with functions:")
    print("MX = ", st.mean(array))
    print("S^2 = ", st.pvariance(array))
    print("D = ", st.variance(array))
    print("sigma = ", st.stdev(array))
    print("A = ", sts.skew(array))
    print("E = ", sts.kurtosis(array))
    print("Me = ", st.median(array))
    print("IQR = ", sts.iqr(array), "\n")


def show_hist_and_pdf_student(array: np.ndarray) -> None:
    ar = array
    ar.sort()
    x = np.linspace(ar[0] - 2, ar[ar.size-1] + 2, 10000)
    y = sts.t.pdf(x, k)
    plt.plot(x, y)
    plt.hist(array, bins="fd", density=True, color="red")
    plt.show()


def find_norm_probability(q=1.45):
    print(sts.norm.cdf(M + q * sg) + 1 - sts.norm.cdf(M - q * sg), "\n")


def find_a_formula(array: np.ndarray, sigma: float, gamma: float) -> None:
    a_mean = array.mean()
    t = sts.norm.ppf((1 + gamma)/2, loc=0, scale=1)
    delta = t * sigma / np.sqrt(array.size)
    x_l = a_mean - delta
    x_r = a_mean + delta
    print("(", x_l, "; ", x_r, ")")


def show_dependence_L_0f_gamma(array: np.ndarray) -> None:
    gm = np.linspace(0, 1, 500)
    t_ar = sts.norm.ppf((gm + 1)/2, loc=0,  scale=1)
    delta_ar = np.array([])
    for i in t_ar:
        delta_ar = np.append(delta_ar, i * sg/np.sqrt(N))
    len_ar = np.array([])
    for i in delta_ar:
        len_ar = np.append(len_ar, array.mean() + i - array.mean() + i)
    plt.plot(gm, len_ar)
    plt.ylabel("L")
    plt.xlabel("gamma")
    plt.title("Рис. 1: Зависимость L от gamma при n = 100")
    plt.show()


def show_dependence_L_of_n(array: np.ndarray) -> None:
    inter = sts.norm.interval(gamma, loc=array.mean(), scale=sg/np.sqrt(N))
    x_ar = np.linspace(5, 5000, dtype=int)
    l_ar = np.array([])
    for i in x_ar:
        ar2 = np.random.normal(loc=M, scale=sg, size=i)
        inter = sts.norm.interval(gamma, loc=ar2.mean(), scale=sg/np.sqrt(i))
        l_ar = np.append(l_ar, inter[1] - inter[0])
    plt.plot(x_ar, l_ar)
    plt.xlabel("Length")
    plt.ylabel("L")
    plt.title("Рис. 2: зависимость L от n")
    plt.show()


def check_inter() -> None:
    M = 5000
    counter = 0
    for i in range(counter):
        array = np.random.normal(loc=M, scale=sg, size=N)
        inter = sts.norm.interval(gamma, loc=array.mean(), scale=sg/np.sqrt(N))
        if inter[0] < M < inter[1]:
            counter += 1
    print("Количество попаданий в доверительный интервал = ", counter/M)
