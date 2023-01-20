import matplotlib.pyplot as plt
import torch
import numpy as np

# Define the loss function
def loss(x):
    return 2.5 + ((x-4).clamp(min=0))**2 + ((-3-x).clamp(min=0))**2 - 2*torch.exp(-x**2) + 0.1*x**2

def plot_lr(alpha=1, title=None):
    # Initialize starting point and step size
    x = torch.tensor([-3.], requires_grad=True)

    # Create an empty list to store the values of x at each step
    x_list = []

    # Perform gradient descent
    for i in range(6):
        x_list.append(x.item())
        y = loss(x)
        y.backward()
        x.data = x.data - alpha*x.grad.data
        x.grad.data.zero_()
    x_list.append(x.item())

    # Plot the loss function
    x_vals = np.linspace(-5, 6, 1000)
    y_vals = loss(torch.from_numpy(x_vals)).detach().numpy()
    plt.plot(x_vals, y_vals)

    # Plot the steps of gradient descent
    plt.scatter(x_list, loss(torch.tensor(x_list)).detach().numpy(), c='r')
    #plt.plot(x_list, loss(torch.tensor(x_list)).detach().numpy(), c='g')

    # Add arrows to show the path of gradient descent
    for i in range(1, len(x_list)):
        plt.arrow(x_list[i-1], loss(torch.tensor([x_list[i-1]])).item(), 
                  x_list[i]-x_list[i-1], loss(torch.tensor([x_list[i]])).item()-loss(torch.tensor([x_list[i-1]])).item(), 
                  head_width=0.25*alpha**0.25, fc='g', ec='g', length_includes_head=True)


    if title is None:
        plt.title(f'LR = {alpha}', fontsize=20)
    else:
        plt.title(title, fontsize=20)
    plt.xlim(-5.1,6.1)
    plt.ylim(0 ,max(y_vals)*1.05)
    plt.xlabel(r"$\omega$", fontsize=16)
    plt.ylabel(r"$\mathcal{L}(\omega$)", fontsize=16)
    plt.savefig(title.lower().replace(' ', '_')+'.svg', format="svg")
    plt.show()

plot_lr(0.5, 'Learning rate too low')
plot_lr(0.8, 'Good learning rate')
plot_lr(3, 'Learning rate too large')
plot_lr(4, 'Learning rate much too large')



####################### Plotting the loss function #######################

def iters_lr(alpha=1, max_it=16):
    # Initialize starting point and step size
    x = torch.tensor([-3.], requires_grad=True)

    # Create an empty list to store the values of x at each step
    x_list = []

    # Perform gradient descent
    for i in range(max_it):
        x_list.append(x.item())
        y = loss(x)
        y.backward()
        x.data = x.data - alpha*x.grad.data
        x.grad.data.zero_()
    y_list = loss(torch.tensor(x_list)).detach().numpy()
    return y_list


from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)

MAX_IT = 25
plt.plot(np.arange(MAX_IT), iters_lr(alpha=0.2, max_it=MAX_IT), label='too low')
plt.plot(np.arange(MAX_IT), iters_lr(alpha=0.7, max_it=MAX_IT), label='good')
plt.plot(np.arange(MAX_IT), iters_lr(alpha=2, max_it=MAX_IT), label='too large')
plt.plot(np.arange(MAX_IT), iters_lr(alpha=10, max_it=MAX_IT), label='way too large')
plt.legend(prop={'size': 16})
plt.xticks(np.arange(MAX_IT))
plt.xlabel(r"Iteration", fontsize=16)
plt.ylim(0,9)
plt.ylabel(r"$\mathcal{L}(\omega$)", fontsize=16)
plt.savefig('loss_per_iter.svg', format="svg")
plt.show()
