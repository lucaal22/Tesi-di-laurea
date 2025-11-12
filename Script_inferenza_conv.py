import os
import torch
import torchvision
import torchvision.transforms as transforms
import ttnn
from loguru import logger
    
    
# Load MNIST data
transform = transforms.Compose([transforms.ToTensor(),     transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

class Autoencoder_conv(torch.nn.Module):

    def __init__(self):
        super(Autoencoder_conv, self).__init__()

        #Encoder
        self.encoder = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, 3),
        torch.nn.ReLU(True),
        torch.nn.Conv2d(4, 8, 3),
        torch.nn.ReLU(True),
        torch.nn.Dropout2d(p=0.1),
                )

        #Decoder
        self.decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(8, 4, 3),
        torch.nn.ReLU(True),
        torch.nn.Dropout2d(p=0.1),
        torch.nn.ConvTranspose2d(4, 1, 3),
        )


    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x



conv_model = Autoencoder_conv()

conv_model.load_state_dict(torch.load("/home/lemnaru_tt/saved_models/conv_model_weights.pt", weights_only=True))

golden_predictions = []

for i, (image, label) in enumerate(testloader):
    if i >= 100:
        break

        #image = image.view(1, -1).to(torch.float32)

    golden_pred = conv_model(image)

    golden_predictions.append(golden_pred)


device = ttnn.open_device(device_id=0, l1_small_size=8192)




if os.path.exists("/home/lemnaru_tt/saved_models/conv_model_weights.pt"):
    weights = torch.load("/home/lemnaru_tt/saved_models/conv_model_weights.pt")
    weights = {
        k: ttnn.from_torch(v, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
        for k, v in weights.items()
    }
    logger.info("Loaded pretrained weights")
else:
    logger.warning("Weights not found, using random weights")
    torch.manual_seed(0)
    weights = {
        "conv1.weight": ttnn.rand((16, 3, 3, 3), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "conv1.bias": ttnn.rand((16,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "conv2.weight": ttnn.rand((32, 16, 3, 3), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "conv2.bias": ttnn.rand((32,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "fc1.weight": ttnn.rand((128, 2048), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "fc1.bias": ttnn.rand((128,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "fc2.weight": ttnn.rand((10, 128), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
        "fc2.bias": ttnn.rand((10,), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device),
    }

def conv_activation_layer(
    input_tensor: ttnn.Tensor,
    input_NHWC: ttnn.Shape,
    conv_outchannels: int,
    weights: dict,
    weight_str: str,
    bias_str: str,
    activation: ttnn.UnaryWithParam,
    device: ttnn.Device,
    conv_kernel_size: tuple,
    conv_stride: tuple = (1, 1),
    conv_padding: tuple = (0, 0),
) -> ttnn.Tensor:
        

        # Extract weight and bias tensors from weights dictionary
    W = weights[weight_str]
    B = weights[bias_str]
    B = ttnn.reshape(B, (1, 1, 1, -1))  # Ensure bias is in correct shape for TT-NN


    # Set up TT-NN convolution configuration including activation function
    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=activation)


    W_on_host = ttnn.from_device(W)
    B_on_host = ttnn.from_device(B)


    preprocessed_W = ttnn.prepare_conv_weights(
        weight_tensor=W_on_host,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.TILE_LAYOUT, #MAI METTERE ROW MAJOR
        weights_format="OIHW",
        in_channels=input_NHWC[3],
        out_channels=conv_outchannels,
        batch_size=input_NHWC[0],
        input_height=input_NHWC[1],
        input_width=input_NHWC[2],
        kernel_size=list(conv_kernel_size),
        stride=list(conv_stride),      
        padding=list(conv_padding),
        dilation=[0,0],
        has_bias=True,
        groups=0,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=activation),
    )

    preprocessed_B = ttnn.prepare_conv_bias(
        bias_tensor=B_on_host,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.TILE_LAYOUT, #MAI METTERE ROW MAJOR
        in_channels=input_NHWC[3],
        out_channels=conv_outchannels,
        batch_size=input_NHWC[0],
        input_height=input_NHWC[1],
        input_width=input_NHWC[2],
        kernel_size=list(conv_kernel_size),
        stride=list(conv_stride),      
        padding=list(conv_padding),
        dilation=[0,0],
        groups=0,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=activation),
    )


    preprocessed_W_on_device = ttnn.to_device(preprocessed_W, device)
    preprocessed_B_on_device = ttnn.to_device(preprocessed_B, device)

    preprocessed_W_on_device = W
    preprocessed_B_on_device = B

        # Perform convolution
    conv_out = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=preprocessed_W_on_device,
        bias_tensor=preprocessed_B_on_device,
        in_channels=input_NHWC[3],
        out_channels=conv_outchannels,
        device=device,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        batch_size=input_NHWC[0],
        input_height=input_NHWC[1],
        input_width=input_NHWC[2],
        conv_config=conv_config,
        groups=0,
    )

    return conv_out

def conv_trans_act_layer(
    input_tensor: ttnn.Tensor,
    input_NHWC: ttnn.Shape,
    conv_outchannels: int,
    weights: dict,
    weight_str: str,
    bias_str: str,
    activation: ttnn.UnaryWithParam,
    device: ttnn.Device,
    conv_kernel_size: tuple,
    conv_stride: tuple = (1, 1),
    conv_padding: tuple = (0, 0),
) -> ttnn.Tensor:
        
        # Extract weight and bias tensors from weights dictionary
    W = weights[weight_str]
    B = weights[bias_str]
    B = ttnn.reshape(B, (1, 1, 1, -1))  # Ensure bias is in correct shape for TT-NN

    conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=activation)

    W_on_host = ttnn.from_device(W)
    B_on_host = ttnn.from_device(B)

    preprocessed_W = ttnn.prepare_conv_transpose2d_weights(
        weight_tensor=W_on_host,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.TILE_LAYOUT, #MAI METTERE ROW MAJOR
        weights_format="OIHW",
        in_channels=input_NHWC[3],
        out_channels=conv_outchannels,
        batch_size=input_NHWC[0],
        input_height=input_NHWC[1],
        input_width=input_NHWC[2],
        kernel_size=list(conv_kernel_size),
        stride=list(conv_stride),      
        padding=list(conv_padding),
        dilation=[0,0],
        has_bias=True,
        groups=0,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=activation),
            )

    preprocessed_B = ttnn.prepare_conv_transpose2d_bias(
        bias_tensor=B_on_host,
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.TILE_LAYOUT, #MAI METTERE ROW MAJOR
        in_channels=input_NHWC[3],
        out_channels=conv_outchannels,
        batch_size=input_NHWC[0],
        input_height=input_NHWC[1],
        input_width=input_NHWC[2],
        kernel_size=list(conv_kernel_size),
        stride=list(conv_stride),      
        padding=list(conv_padding),
        dilation=[0,0],
        groups=0,
        device=device,
        input_dtype=ttnn.bfloat16,
        conv_config = ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=activation),
    )

    preprocessed_W_on_device = ttnn.to_device(preprocessed_W, device)
    preprocessed_B_on_device = ttnn.to_device(preprocessed_B, device)

    conv_trans_out = ttnn.conv_transpose2d(
        input_tensor=input_tensor,
        weight_tensor=preprocessed_W_on_device,
        bias_tensor=preprocessed_B_on_device,
        in_channels=input_NHWC[3],
        out_channels=conv_outchannels,
        device=device,
        kernel_size=conv_kernel_size,
        stride=conv_stride,
        padding=conv_padding,
        batch_size=input_NHWC[0],
        input_height=input_NHWC[1],
        input_width=input_NHWC[2],
        conv_config=conv_config,
        groups=0,
    )

    return conv_trans_out        


dropout_prob = 0.1
tt_predictions=[]

# Run inference on a few test samples
for i, (image, label) in enumerate(testloader):
    if i >=1000:
        break
        # Convert image to TT tensor
    ttnn_image = ttnn.from_torch(image, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_image_permutated = ttnn.permute(ttnn_image, (0, 2, 3, 1))  # NCHW -> NHWC

    #MANCANO I DROPOUT

    #ENCODER

    #Layer 1

    #print(torch.linalg.vector_norm(torch.flatten(ttnn.to_torch(ttnn_image_permutated))))

    conv1_out = conv_activation_layer(
        ttnn_image_permutated, 
        ttnn_image_permutated.shape,
        4,
        weights,
        'encoder.0.weight',
        'encoder.0.bias',
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        device,
        (3, 3),
    )

    #print(torch.linalg.vector_norm(torch.flatten(ttnn.to_torch(conv1_out))))


    #Layer 2

    conv2_out = conv_activation_layer(
        conv1_out,
        (1, 26, 26, 4),
        8,
        weights,
        'encoder.2.weight',
        'encoder.2.bias',
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        device,
        (3, 3),
    )

    #print(torch.linalg.vector_norm(torch.flatten(ttnn.to_torch(conv2_out))))


    #Dropout

    #conv2_drop = ttnn.experimental.dropout(ttnn.to_layout(conv2_out, layout=ttnn.TILE_LAYOUT), probability=dropout_prob, scale=1.0/(1.0 - dropout_prob), seed=0)
    #Decoder

    #Layer 1

    convtr1_out = conv_trans_act_layer(
        conv2_out,
        (1, 24, 24, 8),
        4,
        weights,
        'decoder.0.weight',
        'decoder.0.bias',
        ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        device,
        (3, 3),
    )

    #print(torch.linalg.vector_norm(torch.flatten(ttnn.to_torch(convtr1_out))))


    #convtr1_drop = ttnn.experimental.dropout(ttnn.to_layout(convtr1_out, ttnn.ROW_MAJOR_LAYOUT), probability=dropout_prob, scale=1.0/(1.0 - dropout_prob), seed=0)
        
    convtr2_out = conv_trans_act_layer(
        convtr1_out,
        (1, 26, 26, 4),
        1,
        weights,
        'decoder.3.weight',
        'decoder.3.bias',
        None,
        device,
        (3, 3),
    )
    #print(torch.linalg.vector_norm(torch.flatten(ttnn.to_torch(convtr2_out))))


    # Convert result back to torch)
    prediction = ttnn.to_torch(convtr2_out)
    tt_predictions.append(prediction)

    #logger.info(f"Sample {i+1}: Predicted=N/A, Actual=N/A")

logger.info(f"\nTT-NN SimpleCNN Inference Accuracy: N/A")

ttnn.close_device(device)

#print(golden_predictions[0].shape, tt_predictions[0].shape)

pred_differences = [
    torch.nn.functional.mse_loss(torch.flatten(golden_predictions[i]), torch.flatten(tt_predictions[i])).item()
    for i in range(len(golden_predictions))
]

num_of_buckets = 10
diff_buckets = {i/num_of_buckets: v for (i,v) in enumerate([0] * num_of_buckets)}
diff_buckets['more_than_one'] = 0

for diff in pred_differences:
    if diff/num_of_buckets >= 1.0:
        diff_buckets['more_than_one'] +=1 
    else:        
        diff_buckets[float(int((diff/num_of_buckets) *10)) / 10 ] += 1

#print('Golden: ', golden_predictions)
#print('Tenstorrent: ', tt_predictions)

"""
print('Golden norms')
for p in golden_predictions:
    print(torch.linalg.vector_norm(torch.flatten(p)))

print('TT norms')
for p in tt_predictions:
    print(torch.linalg.vector_norm(torch.flatten(p)))

print('MSE: ', pred_differences)
"""
print('MSE: ', pred_differences[:20])
print(len(golden_predictions), len(tt_predictions))
print(diff_buckets.items())
