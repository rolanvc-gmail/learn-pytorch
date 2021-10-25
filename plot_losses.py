import pandas
import seaborn as sns
import matplotlib.pyplot as plt


myvar = pandas.read_pickle("losses.pkl")
ax = myvar.plot(x="epochs", y="accuracy", legend=False)
ax2 = ax.twinx()
myvar.plot(x="epochs", y="losses", ax=ax2, legend=False, color="r")
ax.figure.legend()
plt.show()
