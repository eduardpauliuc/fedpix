import matplotlib.pyplot as plt

# filename = "inv_results.txt"
filename = "results_fed_20_runs/site-3/results.txt"

with open(filename, 'r') as f:
    results = [x.strip().split(' ') for x in f.readlines()]
    results = [[float(x) for x in y] for y in results]

    # results = results[:-1]

    D_total = [metrics[0] for metrics in results]
    D_real = [metrics[1] for metrics in results]
    D_fake = [metrics[2] for metrics in results]
    G_total = [metrics[3] for metrics in results]
    G_real = [metrics[4] for metrics in results]
    G_fake = [metrics[5] for metrics in results]

    # Create a list of epochs
    epochs = list(range(1, len(results) + 1))

    # Plot the first three metrics (D_total, D_real, D_fake)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, D_total, label='D_total')
    plt.plot(epochs, D_real, label='D_real')
    plt.plot(epochs, D_fake, label='D_fake')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Discriminator Metrics Site 3')
    plt.legend()
    plt.show()

    # Plot the last three metrics (G_total, G_real, G_fake)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, G_total, label='G_total')
    plt.plot(epochs, G_real, label='G_real')
    plt.plot(epochs, G_fake, label='G_fake')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Generator Metrics Site 3')
    plt.legend()
    plt.show()

    # results = [{'d_total': x[0], 'd_real': x[1], 'd_fake': x[2], 'g_total': x[3], 'g_fake': x[4], 'g_real': x[5]} for x
    #            in results]

    # D total: x[0]
    # D real: x[1]
    # D fake: x[2]
    # G total: x[3]
    # G real: x[4]
    # G fake: x[5]
    print(len(results))
