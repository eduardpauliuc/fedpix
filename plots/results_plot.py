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
    plt.title('Discriminator')
    plt.legend()
    plt.show()

    # Plot the last three metrics (G_total, G_real, G_fake)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, G_total, label='G_total')
    plt.plot(epochs, G_real, label='G_real')
    plt.plot(epochs, G_fake, label='G_fake')

    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Generator')
    plt.legend()
    plt.show()

    print(len(results))
