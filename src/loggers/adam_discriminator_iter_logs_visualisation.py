import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/discriminator/good_v1/iteration_logs.csv")
    log_df = pd.read_csv("../gan/adam/adam_logs/adam_88244.5315071/iteration_logs.csv")
    # , iter, error, discriminator real mean error, discriminator fake mean error

    ax = log_df.plot(y=['error', 'discriminator real mean error', 'discriminator fake mean error'])
    ax.legend(['razem', 'próbki rzeczywiste', 'próbki fałszywe'])
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartości błędu dyskryminatora")
    plt.savefig("adam_images/errors.png")
    plt.show()
