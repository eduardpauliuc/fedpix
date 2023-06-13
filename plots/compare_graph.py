import matplotlib.pyplot as plt
import numpy as np
import csv


def get_central():
    filename = "inv_results.txt"

    with open(filename, 'r') as f:
        results = [x.strip().split(' ') for x in f.readlines()]
        results = [[float(x) for x in y] for y in results]
        results = results[:200]
        G_total = [metrics[3] for metrics in results]
        res1 = G_total
    return res1


def get_site():
    filename = "site_local_results.txt"

    with open(filename, 'r') as f:
        results = [x.strip().split(' ') for x in f.readlines()]
        results = [[float(x) for x in y] for y in results]
        results = results[:200]
        G_total = [metrics[3] for metrics in results]
        res1 = G_total
    return res1

def get_fed_gen_mean():
    # filename = "inv_results.txt"
    filename1 = "results_fed_20_runs/site-1/results.txt"
    filename2 = "results_fed_20_runs/site-2/results.txt"
    filename3 = "results_fed_20_runs/site-3/results.txt"

    with open(filename1, 'r') as f:
        results = [x.strip().split(' ') for x in f.readlines()]
        results = [[float(x) for x in y] for y in results]
        results = results[:200]
        G_total = [metrics[3] for metrics in results]
        res1 = G_total

    with open(filename2, 'r') as f:
        results = [x.strip().split(' ') for x in f.readlines()]
        results = [[float(x) for x in y] for y in results]
        results = results[:200]
        G_total = [metrics[3] for metrics in results]
        res2 = G_total

    with open(filename3, 'r') as f:
        results = [x.strip().split(' ') for x in f.readlines()]
        results = [[float(x) for x in y] for y in results]
        results = results[:200]
        G_total = [metrics[3] for metrics in results]
        res3 = G_total

    res = np.dstack([res1, res2, res3])
    res = res.mean(axis=-1)[0]

    return res


def get_feddp_gen_mean():
    filenames = ["results_fedp_20_runs/FL-North-Europe-Site.csv",
                 "results_fedp_20_runs/FL-West-Europe-Site.csv",
                 "results_fedp_20_runs/FL-US-Site.csv"]

    def get_loss(filename):
        with open(filename, 'r') as file:
            reader = csv.reader(file)

            data = []
            for row in reader:
                data.append(row)

            data = data[1:]  # remove headers

            checked = [False for _ in range(200)]
            loss = {}
            for row in data:
                loss[int(row[1])] = float(row[2])
                checked[int(row[1])] = True

            for i in range(len(checked)):
                if not checked[i]:
                    print("Missing ", i)

            return list(loss.values())

    res = [get_loss(filename) for filename in filenames]
    res = np.dstack(res)
    res = res.mean(axis=-1)[0]
    return res


def run():
    res_central = get_central()
    res_fed = get_fed_gen_mean()
    res_dp = get_feddp_gen_mean()
    res_site = get_site()
    # Create a list of epochs
    epochs = list(range(1, len(res_fed) + 1))

    # Plot the first three metrics (D_total, D_real, D_fake)
    plt.figure(figsize=(10, 5))

    plt.plot(epochs, res_central, label='Centralised', color="tab:blue")
    plt.plot(epochs, res_fed, label='Fed', color="tab:green")
    plt.plot(epochs, res_dp, label='Fed + DP', color="tab:red")
    plt.plot(epochs, res_site, label='Site', color="tab:orange")

    # Add vertical lines at every 10 epochs
    for epoch in range(10, len(res_fed) + 1, 10):
        plt.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)

    # Add vertical line to the legend
    plt.plot([], [], color='gray', linestyle='--', alpha=0.3, label='New Round')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Generator Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
