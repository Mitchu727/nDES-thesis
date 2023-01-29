import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/generator/good_v1/iteration_logs.csv")
    log_df = pd.read_csv("../gan/adam/adam_logs/generator/adam_89485.7971838/iteration_logs.csv")

    # , iter, error, discriminator real mean error, discriminator fake mean error
    ax = log_df.plot(y=['error'], legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartości błędu generatora")
    plt.savefig("adam_images/generator_errors.png")
    plt.show()
