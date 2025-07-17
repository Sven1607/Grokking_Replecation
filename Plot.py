from matplotlib import pyplot as plt

# Loss
class plotter:
    def plot(NB_EPOCHS, NB_EPOCH_PLOT, train_loss_history, test_loss_history, train_acc_history, test_acc_history):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NB_EPOCHS + 1), train_loss_history, label='Train Loss')
        plt.plot(range(1, NB_EPOCHS + 1), test_loss_history, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log(Loss)')
        plt.yscale('log')
        plt.legend()
        plt.savefig('loss.png', dpi=300)
        plt.xlim(0,NB_EPOCH_PLOT)
        plt.savefig('loss_zoom.png', dpi=300)


        # Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, NB_EPOCHS + 1), train_acc_history, label='Train Accuracy')
        plt.plot(range(1, NB_EPOCHS + 1), test_acc_history, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy.png', dpi=300)
        plt.xlim(0,NB_EPOCH_PLOT)
        plt.savefig('accuracy_zoom.png', dpi=300)
        plt.show()