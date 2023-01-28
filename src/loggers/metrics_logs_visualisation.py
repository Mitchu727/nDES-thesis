import pandas as pd
import matplotlib.pyplot as plt

# populacja 30000
# generatory populacja
# ndes_1746.6741294
# ndes_20827.0899751 - pretrenowany
#dyskriminator
# ndes_12501.1691769 - pretrenowany
# ndes_25790.0373033

if __name__ == "__main__":
    # ../gan/ndes/ndes_logs
    log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/generator/ndes_59264.1894285/metrics_logs.csv")
# Minimalne wskazanie,Maksymalne wskazanie,Odchylenie standardowe,Funkcja straty

    ax = log_df.plot(y='Średnie wskazanie', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Średnie wskazanie")
    plt.savefig("metrics_images/mean.png")
    plt.show()

    ax = log_df.plot(y='Minimalne wskazanie', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Minimalne wskazanie")
    plt.savefig("metrics_images/min.png")
    plt.show()

    ax = log_df.plot(y='Maksymalne wskazanie', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Maksymalne wskazanie")
    plt.savefig("metrics_images/max.png")
    plt.show()

    ax = log_df.plot(y='Odchylenie standardowe', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Odchylenie standardowe")
    plt.savefig("metrics_images/std.png")
    plt.show()

    ax = log_df.plot(y='Funkcja straty', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Funkcja straty")
    plt.savefig("metrics_images/criterion.png")
    plt.show()

    # ../gan/ndes/ndes_logs
    log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/discriminator/ndes_58595.6661315/metrics_logs.csv")
    # ,Wartość funkcji straty,Średnia predykcja,Błąd dla próbek rzeczywistych,Błąd dla próbek fałszywych
    ax = log_df.plot(y='Wartość funkcji straty', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartość funkcji straty")
    plt.savefig("metrics_images/criterion.png")
    plt.show()

    ax = log_df.plot(y='Średnia predykcja', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Średnia predykcja")
    plt.savefig("metrics_images/mean_prediction.png")
    plt.show()

    ax = log_df.plot(y='Błąd dla próbek rzeczywistych', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Błąd dla próbek rzeczywistych")
    plt.savefig("metrics_images/error_real.png")
    plt.show()

    ax = log_df.plot(y='Błąd dla próbek fałszywych', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Błąd dla próbek fałszywych")
    plt.savefig("metrics_images/error_fake.png")
    plt.show()





    # ax = log_df.plot(x='iter', y='best_fitness', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Najlepsze dopasowanie w populacji")
    # plt.savefig("images/best.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='best_found', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Dopasowanie najlepszego znalezionego osobnika w populacji")
    # plt.savefig("images/best_found.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='pc', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Pęd populacji")
    # plt.savefig("images/pd.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='step', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Wielkość kroku")
    # plt.savefig("images/step.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='fn_cum', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Wartość funkcji celu dla średniej populacji")
    # plt.savefig("images/fn_cum.png")
    # plt.show()