import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    generator_log_df = pd.read_csv("../gan/ndes/ndes_logs/gan/generator/good_v1/metrics_logs.csv")
    discriminator_log_df = pd.read_csv("../gan/ndes/ndes_logs/gan/discriminator/good_v1/metrics_logs.csv")

    # generator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/generator/good_v1/metrics_logs.csv")
    # discriminator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/discriminator/good_v1/metrics_logs.csv")
    #
    # generator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/generator/good_v1/metrics_logs.csv")
    # discriminator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/discriminator/good_v1/metrics_logs.csv")

    # Minimalne wskazanie,Maksymalne wskazanie,Odchylenie standardowe,Funkcja straty
    ax = generator_log_df.plot(y=['Maksymalne wskazanie', 'Średnie wskazanie', 'Minimalne wskazanie'], legend=None)
    ax.legend(['maksymalne', 'średnie', 'minimalne'])
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wskazanie dyskryminatora dla generowanych obrazów")
    plt.savefig("metrics_images/mean.pdf")
    plt.show()

    ax = generator_log_df.plot(y='Odchylenie standardowe', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Odchylenie standardowe")
    plt.savefig("metrics_images/std.pdf")
    plt.show()

    ax = generator_log_df.plot(y='Funkcja straty', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartość funkcji straty generatora")
    plt.savefig("metrics_images/generator_criterion.pdf")
    plt.show()

    # ,Wartość funkcji straty,Średnia predykcja,Błąd dla próbek rzeczywistych,Błąd dla próbek fałszywych
    ax = discriminator_log_df.plot(y='Wartość funkcji straty', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartość funkcji straty dyskryminatora")
    plt.savefig("metrics_images/discriminator_criterion.pdf")
    plt.show()

    ax = discriminator_log_df.plot(y='Średnia predykcja', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Średnia predykcja dyskryminatora")
    plt.savefig("metrics_images/mean_prediction.pdf")
    plt.show()

    ax = discriminator_log_df.plot(y='Błąd dla próbek rzeczywistych', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Błąd dla próbek rzeczywistych")
    plt.savefig("metrics_images/error_real.pdf")
    plt.show()

    ax = discriminator_log_df.plot(y='Błąd dla próbek fałszywych', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Błąd dla próbek fałszywych")
    plt.savefig("metrics_images/error_fake.pdf")
    plt.show()
