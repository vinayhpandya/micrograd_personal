# micrograd_personal
This is an implementation of Andrej Karpathy's Micrograd video for complete backpropagation in neural networks
I'll add some more experiments with sklearn datasets

Some experiments with Micrograd
```
from sklearn.datasets import make_moons, make_blobs

X, y = make_moons(n_samples=100, noise=0.1)

y = y * 2 - 1  # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap="jet")
```
![](images/output.png)

After training for 100 epochs/episodes we get 100% accuracy
```
for k in range(100):
    # forward
    total_loss, acc = loss()

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.9 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
```
![](images/demo.png)