import torch
import torch.nn as nn
import os
import Prepdata
import Model_Definition
import devices
import Plot

NB_EPOCHS = 80
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 1

Prepdata.Prep.set_seed(Prepdata.Prep.SEED)
model = Model_Definition.EmbeddingConcatFFModel().to(devices.device.device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Recording
REPORT_INTERVAL = 20    # How often we print
SAVE_INTERVAL = 50      # How often we save
model_folder = 'intermediary'
# Create folder if it doesn't exist
if not os.path.exists(model_folder):
    os.makedirs(model_folder)

train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

for epoch in range(NB_EPOCHS):
    
    # Training phase
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch in Prepdata.Prep.train_loader:
        x1, x2, y = batch[:,0], batch[:,1], batch[:,2]
        optimizer.zero_grad()
        output = model(x1,x2)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (output.argmax(dim=1) == y).sum().item()
        
    train_loss /= len(Prepdata.Prep.train_loader)
    train_loss_history.append(train_loss)
    train_acc /= len(Prepdata.Prep.train_data)
    train_acc_history.append(train_acc)
    
    # Testing phase
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        test_acc = 0.0
        for batch in Prepdata.Prep.test_loader:
            x1, x2, y = batch[:,0], batch[:,1], batch[:,2]
            output = model(x1,x2)
            loss = criterion(output, y)
            test_loss += loss.item()
            test_acc += (output.argmax(dim=1) == y).sum().item()
        
        test_loss /= len(Prepdata.Prep.test_loader)
        test_loss_history.append(test_loss)
        test_acc /= len(Prepdata.Prep.test_data)
        test_acc_history.append(test_acc)
        
    if epoch % REPORT_INTERVAL == 0:
        print(f"{epoch}/{NB_EPOCHS}: Train loss={train_loss:.4f}, acc={100*train_acc:.1f}%  /  Test loss={test_loss:.4f}, acc={100*test_acc:.1f}%")
    
    if epoch % SAVE_INTERVAL == 0:
        # Save model in intermediary folder
        torch.save(model.state_dict(), f"{model_folder}/model_{epoch}.pth")
        
torch.save(model.state_dict(), f"model.pth")

Plot.plotter.plot(NB_EPOCHS = NB_EPOCHS, NB_EPOCH_PLOT=3000, train_loss_history=train_loss_history, test_loss_history=test_loss_history, train_acc_history=train_acc_history, test_acc_history=test_acc_history)