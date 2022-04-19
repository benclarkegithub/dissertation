# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms

import blitz

batch_size = 16

transform = transforms.Compose(
    [transforms.ToTensor()]
)

test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

classes = [str(x) for x in range(10)]

data_iter = iter(test_loader)
images, labels = data_iter.next()

# blitz_helper.show_image(torchvision.utils.make_grid(images))
print(f"True labels: ", " ".join(f"{classes[labels[i]]}" for i in range(batch_size)))

net = blitz.Net()
net.load_state_dict(torch.load("./MNIST_net.pth"))
output = net(images)

_, predicted = torch.max(output, 1)
print(f"Predicted labels: ", " ".join(f"{classes[predicted[i]]}" for i in range(batch_size)))

# Copied and pasted
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
