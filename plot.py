import matplotlib.pyplot as plt

# Data: accuracy for each epoch
epochs = list(range(1, 25))
accuracy = [
    0.2623, 0.2883, 0.2836, 0.2849, 0.2864, 0.2872, 0.2874, 0.2863,
    0.2898, 0.2864, 0.2898, 0.2919, 0.2908, 0.2909, 0.2912, 0.2923,
    0.2928, 0.2904, 0.2928, 0.2914, 0.2921, 0.2926, 0.2938, 0.2935
]

plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracy, marker='o', linestyle='-', color='b')
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
plt.grid(True)
plt.show()
