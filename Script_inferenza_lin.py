import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import ttnn
import os
from loguru import logger


# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(),     transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

#Modello GPU per il confronto

class Autoencoder_linear(torch.nn.Module):

    def __init__(self):
        super(Autoencoder_linear, self).__init__()

        #Encoder
        self.encoder = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 500),
        torch.nn.ReLU(True),
        torch.nn.Linear(500, 300),
        torch.nn.ReLU(True),
        torch.nn.Linear(300, 100),
        torch.nn.ReLU(True),
        torch.nn.Linear(100, 40)
        )

        #Decoder
        self.decoder = torch.nn.Sequential(
        torch.nn.Linear(40, 300),
        torch.nn.ReLU(True),
        torch.nn.Linear(300, 784),
        torch.nn.Unflatten(1, (1, 28, 28))
        )


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

linear_model = Autoencoder_linear()

linear_model.load_state_dict(torch.load("/home/lemnaru_tt/saved_models/linear_model_weights.pt", weights_only=True))

golden_predictions = []

for i, (image, label) in enumerate(testloader):
    if i >= 5:
        break

    image = image.view(1, -1).to(torch.float32)

    golden_pred = linear_model(image)

    golden_predictions.append(golden_pred)




#Script di inferenza del modello lineare


# Open Tenstorrent device
device = ttnn.open_device(device_id=0)




if os.path.exists("/home/lemnaru_tt/saved_models/linear_model_weights.pt"):
# Pretrained weights
    weights = torch.load("/home/lemnaru_tt/saved_models/linear_model_weights.pt")

    linE_W1 = ttnn.from_torch(weights["encoder.1.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_b1 = ttnn.from_torch(weights["encoder.1.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_W2 = ttnn.from_torch(weights["encoder.3.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_b2 = ttnn.from_torch(weights["encoder.3.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_W3 = ttnn.from_torch(weights["encoder.5.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_b3 = ttnn.from_torch(weights["encoder.5.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_W4 = ttnn.from_torch(weights["encoder.7.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linE_b4 = ttnn.from_torch(weights["encoder.7.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        
    linD_W1 = ttnn.from_torch(weights["decoder.0.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linD_b1 = ttnn.from_torch(weights["decoder.0.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linD_W2 = ttnn.from_torch(weights["decoder.2.weight"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    linD_b2 = ttnn.from_torch(weights["decoder.2.bias"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    logger.info("Loaded pretrained weights from linear_model_weights.pt")
else:
    raise ValueError('Weights directory not found')

tt_predictions = []

for i, (image, label) in enumerate(testloader):
    if i >= 5:
        break

    image = image.view(1, -1).to(torch.float32)
    image_tt = ttnn.from_torch(image, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


    #ENCODER

    # Layer 1

    lEW1_final = ttnn.transpose(linE_W1, -2, -1)
    lEb1_final = ttnn.reshape(linE_b1, [1, -1])
    lE_out1 = ttnn.linear(image_tt, lEW1_final, bias=lEb1_final)
    lE_out1 = ttnn.relu(lE_out1)


    # Layer 2

    lEW2_final = ttnn.transpose(linE_W2, -2, -1)
    lEb2_final = ttnn.reshape(linE_b2, [1, -1])
    lE_out2 = ttnn.linear(lE_out1, lEW2_final, bias=lEb2_final)
    lE_out2 = ttnn.relu(lE_out2)


    # Layer 3

    lEW3_final = ttnn.transpose(linE_W3, -2, -1)
    lEb3_final = ttnn.reshape(linE_b3, [1, -1])
    lE_out3 = ttnn.linear(lE_out2, lEW3_final, bias=lEb3_final)
    lE_out3 = ttnn.relu(lE_out3)


    #Layer 4

    lEW4_final = ttnn.transpose(linE_W4, -2, -1)
    lEb4_final = ttnn.reshape(linE_b4, [1, -1])
    lE_out4 = ttnn.linear(lE_out3, lEW4_final, bias=lEb4_final)


    #DECODER

    #Layer 1

    lDW1_final = ttnn.transpose(linD_W1, -2, -1)
    lDb1_final = ttnn.reshape(linD_b1, [1, -1])
    lD_out1 = ttnn.linear(lE_out4, lDW1_final, bias=lDb1_final)
    lD_out1 = ttnn.relu(lD_out1)


    #Layer 2

    lDW2_final = ttnn.transpose(linD_W2, -2, -1)
    lDb2_final = ttnn.reshape(linD_b2, [1, -1])
    lD_out2 = ttnn.linear(lD_out1, lDW2_final, bias=lDb2_final)
    lD_out2 = ttnn.relu(lD_out2)


    # Convert result back to torch
    prediction = ttnn.to_torch(lD_out2)
    tt_predictions.append(prediction)
    #RICORDA di metterlo nella forma giusta, qui dovrebbe essere ancora flattenato


    #Qui devi ritornare errore (MSE?) tra input e ricostruzione, oppure lo printi ma non so se è fattibile,
    #oppure lo salvi e poi lo visualizzi

    logger.info(f"Sample {i+1}: Predicted=N/A, Actual=N/A")

logger.info(f"\nInference Accuracy: N/A ")

ttnn.close_device(device)

print(golden_predictions[0].shape, tt_predictions[0].shape)


pred_differences = [
    torch.nn.functional.mse_loss(torch.flatten(golden_predictions[i]), torch.flatten(tt_predictions[i]))
    for i in range(len(golden_predictions))
]

#print('Golden: ', golden_predictions)
#print('Tenstorrent: ', tt_predictions)

print('Golden norms')
for p in golden_predictions:
    print(torch.linalg.vector_norm(torch.flatten(p)))

print('TT norms')
for p in tt_predictions:
    print(torch.linalg.vector_norm(torch.flatten(p)))

print('MSE: ', pred_differences)

    