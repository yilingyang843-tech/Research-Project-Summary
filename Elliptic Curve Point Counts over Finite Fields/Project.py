import os
import math
import time
import pandas as pd
import matplotlib.pyplot as plt


FIGURE_DIR = "figures"
TABLE_DIR = "tables"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["figure.dpi"] = 120

def is_prime(n: int) -> bool:
    if n < 2:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    limit = int(math.sqrt(n)) + 1

    for d in range(3, limit, 2):
        if n % d == 0:
            return False

    return True


def primes_less_than(N: int):
    return [n for n in range(2, N) if is_prime(n)]


def legendre_symbol(a: int, p: int) -> int:
    a = a % p

    if a == 0:
        return 0

    value = pow(a, (p - 1) // 2, p)

    if value == 1:
        return 1

    if value == p - 1:
        return -1

    raise ValueError("Unexpected value in Legendre symbol computation.")


def curve_rhs(x: int, a: int, b: int, p: int) -> int:
    return (x**3 + a * x + b) % p


def is_nonsingular(a: int, b: int, p: int) -> bool:
    return (4 * a**3 + 27 * b**2) % p != 0


def point_count_direct(a: int, b: int, p: int) -> int:
    count = 0

    for x in range(p):
        rhs = curve_rhs(x, a, b, p)

        for y in range(p):
            if (y * y - rhs) % p == 0:
                count += 1

    return count + 1


def point_count_legendre(a: int, b: int, p: int) -> int:
    character_sum = 0

    for x in range(p):
        rhs = curve_rhs(x, a, b, p)
        character_sum += legendre_symbol(rhs, p)

    return p + 1 + character_sum


def frobenius_trace(point_count: int, p: int) -> int:
    return p + 1 - point_count


def normalized_trace(ap: int, p: int) -> float:
    return ap / (2 * math.sqrt(p))


curves = {
    "E1": {
        "label": r"$E_1: y^2 = x^3 + 1$",
        "a": 0,
        "b": 1,
        "hist_file": "Figure_3_Histogram_E1.png"
    },
    "E2": {
        "label": r"$E_2: y^2 = x^3 - x$",
        "a": -1,
        "b": 0,
        "hist_file": "Figure_4_Histogram_E2.png"
    },
    "E3": {
        "label": r"$E_3: y^2 = x^3 + x + 1$",
        "a": 1,
        "b": 1,
        "hist_file": "Figure_5_Histogram_E3.png"
    }
}

P_MAX = 500
prime_list = primes_less_than(P_MAX)

# Legendre symbol is defined for odd primes, so exclude p = 2.
prime_list = [p for p in prime_list if p != 2]

all_results = []

for curve_key, params in curves.items():
    a = params["a"]
    b = params["b"]
    curve_label = params["label"]

    for p in prime_list:
        if not is_nonsingular(a, b, p):
            continue

        point_count = point_count_legendre(a, b, p)
        ap = frobenius_trace(point_count, p)
        norm_ap = normalized_trace(ap, p)

        all_results.append({
            "curve": curve_key,
            "curve_equation": curve_label,
            "a": a,
            "b": b,
            "p": p,
            "point_count": point_count,
            "frobenius_trace_ap": ap,
            "normalized_trace": norm_ap,
            "lower_hasse_bound": -2 * math.sqrt(p),
            "upper_hasse_bound": 2 * math.sqrt(p)
        })

df = pd.DataFrame(all_results)

df.to_csv(
    os.path.join(TABLE_DIR, "elliptic_curve_trace_results.csv"),
    index=False
)

print("Saved: tables/elliptic_curve_trace_results.csv")
print(df.head())


plt.figure(figsize=(11, 7))

for curve_key, params in curves.items():
    sub = df[df["curve"] == curve_key]

    plt.plot(
        sub["p"],
        sub["frobenius_trace_ap"],
        marker="o",
        markersize=5,
        linewidth=2.0,
        label=params["label"]
    )

p_values = sorted(df["p"].unique())
upper_bound = [2 * math.sqrt(p) for p in p_values]
lower_bound = [-2 * math.sqrt(p) for p in p_values]

plt.plot(
    p_values,
    upper_bound,
    linestyle="--",
    linewidth=2.2,
    label=r"$2\sqrt{p}$"
)

plt.plot(
    p_values,
    lower_bound,
    linestyle="--",
    linewidth=2.2,
    label=r"$-2\sqrt{p}$"
)

plt.xlabel("Prime $p$", fontweight="bold")
plt.ylabel(r"Frobenius Trace $a_p$", fontweight="bold")
plt.title(r"Frobenius Traces $a_p$ for Three Elliptic Curves", fontweight="bold")
plt.legend(frameon=True)
plt.grid(True, alpha=0.35, linewidth=0.8)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "Figure_1_Frobenius_Trace.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: figures/Figure_1_Frobenius_Trace.png")


plt.figure(figsize=(11, 7))

for curve_key, params in curves.items():
    sub = df[df["curve"] == curve_key]

    plt.plot(
        sub["p"],
        sub["normalized_trace"],
        marker="o",
        markersize=5,
        linewidth=2.0,
        label=params["label"]
    )

plt.axhline(
    1,
    linestyle="--",
    linewidth=2.0,
    label=r"Upper bound $=1$"
)

plt.axhline(
    -1,
    linestyle="--",
    linewidth=2.0,
    label=r"Lower bound $=-1$"
)

plt.axhline(
    0,
    linestyle="-",
    linewidth=1.4
)

plt.xlabel("Prime $p$", fontweight="bold")
plt.ylabel(r"Normalized Trace $a_p/(2\sqrt{p})$", fontweight="bold")
plt.title(r"Normalized Frobenius Traces for Three Elliptic Curves", fontweight="bold")
plt.legend(frameon=True)
plt.grid(True, alpha=0.35, linewidth=0.8)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "Figure_2_Normalized_Frobenius_Trace.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved: figures/Figure_2_Normalized_Frobenius_Trace.png")

for curve_key, params in curves.items():
    sub = df[df["curve"] == curve_key]

    plt.figure(figsize=(9, 6))

    plt.hist(
        sub["normalized_trace"],
        bins=20,
        edgecolor="black",
        linewidth=1.1,
        alpha=0.8
    )

    plt.axvline(0, linestyle="--", linewidth=2.0)
    plt.axvline(1, linestyle="--", linewidth=2.0)
    plt.axvline(-1, linestyle="--", linewidth=2.0)

    plt.xlabel(r"Normalized Trace $a_p/(2\sqrt{p})$", fontweight="bold")
    plt.ylabel("Frequency", fontweight="bold")
    plt.title(
        f"Histogram of Normalized Frobenius Traces for {params['label']}",
        fontweight="bold"
    )

    plt.grid(True, alpha=0.35, linewidth=0.8)
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURE_DIR, params["hist_file"]),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved: figures/{params['hist_file']}")

runtime_results = []
runtime_prime_limits = [50, 100, 200, 300, 500]

for limit in runtime_prime_limits:
    selected_primes = [p for p in prime_list if p < limit]

    for curve_key, params in curves.items():
        a = params["a"]
        b = params["b"]

        selected_valid_primes = [
            p for p in selected_primes
            if is_nonsingular(a, b, p)
        ]

        start_direct = time.perf_counter()

        for p in selected_valid_primes:
            point_count_direct(a, b, p)

        end_direct = time.perf_counter()
        direct_time = end_direct - start_direct

        start_legendre = time.perf_counter()

        for p in selected_valid_primes:
            point_count_legendre(a, b, p)

        end_legendre = time.perf_counter()
        legendre_time = end_legendre - start_legendre

        speedup = direct_time / legendre_time if legendre_time > 0 else float("inf")

        runtime_results.append({
            "curve": curve_key,
            "prime_range": f"p < {limit}",
            "number_of_primes": len(selected_valid_primes),
            "direct_enumeration_time_seconds": direct_time,
            "legendre_method_time_seconds": legendre_time,
            "speedup": speedup
        })

runtime_df = pd.DataFrame(runtime_results)

runtime_df["direct_enumeration_time_seconds"] = runtime_df[
    "direct_enumeration_time_seconds"
].round(6)

runtime_df["legendre_method_time_seconds"] = runtime_df[
    "legendre_method_time_seconds"
].round(6)

runtime_df["speedup"] = runtime_df["speedup"].round(2)

runtime_df.to_csv(
    os.path.join(TABLE_DIR, "runtime_comparison_table.csv"),
    index=False
)

print("\nSaved: tables/runtime_comparison_table.csv")
print(runtime_df)


agreement_results = []
check_primes = [p for p in prime_list if p < 100]

for curve_key, params in curves.items():
    a = params["a"]
    b = params["b"]

    for p in check_primes:
        if not is_nonsingular(a, b, p):
            continue

        direct_count = point_count_direct(a, b, p)
        legendre_count = point_count_legendre(a, b, p)

        agreement_results.append({
            "curve": curve_key,
            "p": p,
            "direct_count": direct_count,
            "legendre_count": legendre_count,
            "agree": direct_count == legendre_count
        })

agreement_df = pd.DataFrame(agreement_results)

agreement_df.to_csv(
    os.path.join(TABLE_DIR, "method_agreement_check.csv"),
    index=False
)

print("\nSaved: tables/method_agreement_check.csv")
print(agreement_df.head())

if agreement_df["agree"].all():
    print("\nAll checked cases agree.")
else:
    print("\nWarning: Some cases do not agree.")


summary_stats = df.groupby("curve")["normalized_trace"].agg(
    count="count",
    mean="mean",
    std="std",
    min="min",
    max="max"
).reset_index()

summary_stats["mean"] = summary_stats["mean"].round(6)
summary_stats["std"] = summary_stats["std"].round(6)
summary_stats["min"] = summary_stats["min"].round(6)
summary_stats["max"] = summary_stats["max"].round(6)

summary_stats.to_csv(
    os.path.join(TABLE_DIR, "normalized_trace_summary_statistics.csv"),
    index=False
)

print("\nSaved: tables/normalized_trace_summary_statistics.csv")
print(summary_stats)
