import numpy as np

def calculate_accuracies(confusion_matrix, epoch):
    num_classes = len(confusion_matrix)
    class_accuracies = []

    # Calculate accuracy per class
    for i in range(num_classes):
        correct = confusion_matrix[i, i]  # True positives for class i
        total = np.sum(confusion_matrix[i, :])  # Total samples for class i
        accuracy = correct / total if total > 0 else 0  # Avoid division by zero
        class_accuracies.append((correct, total, accuracy))

    # Calculate total accuracy
    total_correct = np.trace(confusion_matrix)  # Sum of diagonal elements
    total_samples = np.sum(confusion_matrix)  # Total samples
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Build a summary string for class-wise accuracies
    accuracies_summary = ", ".join(
        [f"Class {i}: {acc[2]:.2%} ({int(acc[0])}/{int(acc[1])})" for i, acc in enumerate(class_accuracies)]
    )

    # Print the results
    print(f"Epoch {epoch}: {accuracies_summary}, Total Accuracy: {total_accuracy: .2%}.")
