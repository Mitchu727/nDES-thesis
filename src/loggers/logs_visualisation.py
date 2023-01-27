import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    log_df = pd.read_csv("../gan/ndes/ndes_logs/discriminator/ndes_36722.4028387/iteration_logs.csv")

    ax = log_df.plot(x='iter', y='mean_fitness', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Średnie dopasowanie populacji")
    plt.savefig("images/mean.png")
    plt.show()

    ax = log_df.plot(x='iter', y='best_fitness', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Najlepsze dopasowanie w populacji")
    plt.savefig("images/best.png")
    plt.show()

    ax = log_df.plot(x='iter', y='best_found', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Dopasowanie najlepszego znalezionego osobnika w populacji")
    plt.savefig("images/best_found.png")
    plt.show()

    ax = log_df.plot(x='iter', y='pc', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Pęd populacji")
    plt.savefig("images/pd.png")
    plt.show()

    ax = log_df.plot(x='iter', y='step', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wielkość kroku")
    plt.savefig("images/step.png")
    plt.show()

    ax = log_df.plot(x='iter', y='fn_cum', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartość funkcji celu dla średniej populacji")
    plt.savefig("fn_cum.png")
    plt.show()