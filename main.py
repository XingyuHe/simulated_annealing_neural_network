from model import *
import matplotlib.pyplot as plt
import pandas as pd

m = Trainer()
acceptance_series, accuracy_series, loss_series = m.load_train_record()
train_accuracy, test_accuracy = accuracy_series[:, 0], accuracy_series[:, 1]
m.plot(train_accuracy)
m.plot(test_accuracy)
loss_series = loss_series[1:]

df = pd.DataFrame({"acceptance":acceptance_series,
                   "train_accuracy":train_accuracy,
                   "test_accuracy":test_accuracy,
                   "loss":loss_series})
accuracy_df = pd.DataFrame({
                   "train_accuracy":train_accuracy,
                   "test_accuracy":test_accuracy})

df.cumsum()
df.plot()

accuracy_df.cumsum()
plt.figure()
accuracy_df.plot(style="k--")

loss_df = df["loss"]
loss_df.cumsum()
loss_df.plot(); plt.legend()

acceptance_df = df["acceptance"]
cum_acceptance_df = acceptance_df.cumsum()
cum_acceptance_df.plot()
plt.legend(["cumulative acceptance"])
