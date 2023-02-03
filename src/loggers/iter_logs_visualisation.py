import pandas as pd
import matplotlib.pyplot as plt


#dyskriminator
# ndes_25790.0373033
# ndes_12501.1691769 - pretrenowany
# generatory
# ndes_1746.6741294
# ndes_20827.0899751 - pretrenowany


if __name__ == "__main__":

    # log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/generator/good_v1/iteration_logs.csv")
    # log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/discriminator/good_v1/iteration_logs.csv")
    # log_df = pd.read_csv("../../ndes_logs/generator/ndes_227.3185525/iteration_logs.csv")
    # log_df = pd.read_csv("../../ndes_logs/generator/ndes_227.3185525/iteration_logs.csv")


    # dyskriminator
    # log_df = pd.read_csv("../../ndes_logs/discriminator/ndes_25790.0373033/iteration_logs.csv")
    # log_df = pd.read_csv("../../ndes_logs/discriminator/ndes_12501.1691769/iteration_logs.csv")  # pretrenowany
    # generator
    # log_df = pd.read_csv("../../ndes_logs/generator/ndes_227.3185525/iteration_logs.csv")
    # log_df = pd.read_csv("../../ndes_logs/generator/ndes_20827.0899751/iteration_logs.csv")  # pretrenowany

    # log_df = pd.read_csv("../gan/ndes/ndes_logs/gan/discriminator/good_v1/iteration_logs.csv")
    # log_df = pd.read_csv("../gan/ndes/ndes_logs/gan/generator/good_v1/iteration_logs.csv")
    #
    # log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/generator/good_v1/iteration_logs.csv")
    log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/discriminator/good_v1/iteration_logs.csv")


    ax = log_df.plot(y='mean_fitness', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Średnie dopasowanie populacji")
    plt.savefig("iter_images/mean.pdf")
    plt.show()

    ax = log_df.plot(y=['best_fitness', 'best_found'], legend=None)
    ax.legend(["w populacji", "dotychczas znalezione"])
    plt.grid(True)
    plt.title("Najlepsze dopasowanie")
    plt.savefig("iter_images/best.pdf")
    plt.show()

    ax = log_df.plot(y='step', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wielkość kroku")
    plt.savefig("iter_images/step.pdf")
    plt.show()


    # ax = log_df.plot(y='pc', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Pęd populacji")
    # plt.savefig("iter_images/pd.png")
    # plt.show()
    #
    #
    # ax = log_df.plot(y='fn_cum', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Wartość funkcji celu dla średniej populacji")
    # plt.savefig("iter_images/fn_cum.png")
    # plt.show()