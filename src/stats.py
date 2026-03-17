import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

df = pd.read_csv("results/history.csv")
df["time"] = pd.to_datetime(df["time"])

for word in df["word"].unique():
    sub = df[df["word"] == word]
    plt.plot(sub["time"], sub["pron"], label=word)

plt.legend()
plt.title("Progress")
plt.show()