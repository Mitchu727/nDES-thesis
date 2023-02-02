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
    # log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/generator/ndes_59264.1894285/metrics_logs.csv")

    # generator_log_df = pd.read_csv("../gan/ndes/ndes_logs/gan/generator/good_v1/metrics_logs.csv")
    # discriminator_log_df = pd.read_csv("../gan/ndes/ndes_logs/gan/discriminator/good_v1/metrics_logs.csv")

    # generator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/generator/good_v1/metrics_logs.csv")
    # discriminator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan/discriminator/good_v1/metrics_logs.csv")

    generator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/generator/good_v1/metrics_logs.csv")
    discriminator_log_df = pd.read_csv("../gan/mixed/mixed_logs/gan_reversed/discriminator/good_v1/metrics_logs.csv")

    # ax = log_df.plot(y=['error', 'discriminator real mean error', 'discriminator fake mean error'])
    # ax.legend(['razem', 'próbki rzeczywiste', 'próbki fałszywe'])

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
    plt.title("Wartość funkcji straty dyskryminatora")
    plt.savefig("metrics_images/discriminator_criterion.pdf")
    plt.show()

    # ,Wartość funkcji straty,Średnia predykcja,Błąd dla próbek rzeczywistych,Błąd dla próbek fałszywych
    ax = discriminator_log_df.plot(y='Wartość funkcji straty', legend=None)
    ax.xaxis.set_label_text("")
    plt.grid(True)
    plt.title("Wartość funkcji straty generatora")
    plt.savefig("metrics_images/generator_criterion.pdf")
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





    # ax = log_df.plot(x='iter', y='best_fitness', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Najlepsze dopasowanie w populacji")
    # plt.savefig("iter_images/best.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='best_found', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Dopasowanie najlepszego znalezionego osobnika w populacji")
    # plt.savefig("iter_images/best_found.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='pc', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Pęd populacji")
    # plt.savefig("iter_images/pd.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='step', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Wielkość kroku")
    # plt.savefig("iter_images/step.png")
    # plt.show()
    #
    # ax = log_df.plot(x='iter', y='fn_cum', legend=None)
    # ax.xaxis.set_label_text("")
    # plt.grid(True)
    # plt.title("Wartość funkcji celu dla średniej populacji")
    # plt.savefig("iter_images/fn_cum.png")
    # plt.show()