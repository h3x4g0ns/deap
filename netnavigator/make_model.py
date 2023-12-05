import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Adam
import math
import argparse
import pprint

iteration_count = 0
device = torch.device("cpu")

class EmbeddingLayer(nn.Embedding):
    def __init__(self, ntoken, ninp, initrange):
        super().__init__(ntoken, ninp)
        self.ninp = ninp
        nn.init.uniform_(self.weight, -initrange, initrange)

    def forward(self, src):
        return super().forward(src) * math.sqrt(self.ninp)

class PositionalEncodingLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class TransformerDecoderLayer(nn.TransformerEncoderLayer):
    """Though this class inherits from torch.nn.TransformerEncoderLayer,
    it functions as a decoder in this model"""

    def __init__(self, ninp, nhead, nhid, droupout):
        super().__init__(ninp, nhead, nhid, droupout)
        self.src_mask = None

    def forward(self, src):
        global iteration_count
        iteration_count += 1

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        return super().forward(src, self.src_mask)

class LinearLayer(nn.Linear):
    def __init__(self, ninp, ntoken, initrange):
        super().__init__(ninp, ntoken)
        nn.init.zeros_(self.bias)
        nn.init.uniform_(self.weight, -initrange, initrange)

class TransformerLMSequential(nn.Sequential):
    """A small language model based on the design of GPT-2 using nn.Sequential
    for compatibility with Pipe"""

    def __init__(self, ntokens, ninp, nhead, nhid, dropout, initrange, ndecoder):
        layers = [
            EmbeddingLayer(ntokens, ninp, initrange),
            PositionalEncodingLayer(ninp, dropout),
        ]
        for _ in range(ndecoder):
            layers.append(TransformerDecoderLayer(ninp, nhead, nhid, dropout))

        layers.append(LinearLayer(ninp, ntokens, initrange))
        super().__init__(*layers)

def make_model(args):
    """
    Builds a sequential model to be partitioned based on the input hyperparameters
    """
    ninp = args.embedding_dimension  # embedding dimension
    nhid = (args.forward_dimension)  # dimension of the feedforward network model in nn.TransformerEncoder
    nhead = args.num_heads  # num of heads in the multiheadattention models
    dropout = args.dropout
    initrange = args.init_range
    ndecoder = args.num_decoder_layers

    model = TransformerLMSequential(
        args.vocab_size, ninp, nhead, nhid, dropout, initrange, ndecoder
    ).to(device)
    return model

def generate_balance(num_devices, num_layers):
    """
    Distribute layers across multiple devices, returns the distribution of
    layers across the devices.
    """
    balance = []
    layers_assigned = 0
    for i in range(num_devices):
        x = (num_layers - layers_assigned) / (num_devices - i)
        if x.is_integer():
            balance.append(int(x))
            layers_assigned += x
        else:
            balance.append(math.ceil(x))
            layers_assigned += math.ceil(x)
    return balance

def extract_layer_info(layer):
    """
    Recursively extract layer information, handling layers with nested sub-layers.
    Ignoring Dropout layers and returning input and output features for each layer.
    """
    sub_layers = list(layer.named_children())
    layer_infos = []
    layer_info = {}
    if sub_layers:
        for sub_layer_name, sub_layer in sub_layers:
            # layers to ignore
            if isinstance(sub_layer, nn.Dropout):
                continue
            layer_infos.extend(extract_layer_info(sub_layer))
    else:
        layer_info['type'] = layer.__class__.__name__
        if isinstance(layer, nn.Linear):
            layer_info['input_features'] = layer.in_features
            layer_info['output_features'] = layer.out_features
            layer_info['bias_terms'] = layer.bias.nelement() if layer.bias is not None else 0
        elif isinstance(layer, nn.Embedding):
            layer_info['input_features'] = layer.num_embeddings
            layer_info['output_features'] = layer.embedding_dim
    if layer_info:
        layer_infos.append(layer_info)
    return layer_infos

def make_partition(model, args):
    """
    Given sequential model, splits up layers
    return partition 
    """
    partition_info = []
    for layer in model:
        layer_info = extract_layer_info(layer)
        if layer_info:
            partition_info.append(layer_info)
    return partition_info
    

def runner(args):
    model = make_model(args).eval()
    partition = make_partition(model, args)
    pprint.pprint(partition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build parallelizable LLM")
    parser.add_argument("--embedding-dimension", type=int, default=2048, help="encoding embedding dimension")
    parser.add_argument("--forward-dimension", type=int, default=2048, help="nn.TransformerEncoder feedforward input dimension")
    parser.add_argument("--num-heads", type=int, default=32, help="number of heads in multiheaded attention models")
    parser.add_argument("--dropout", type=float, default=0, help="nn dropout")
    parser.add_argument("--init-range", type=float, default=0.1, help="tensor init range")
    parser.add_argument("--num-decoder-layers", type=int, default=10, help="number of decoder layers")
    parser.add_argument("--num-devices", type=int, default=2, help="number of devices")
    parser.add_argument("--vocab-size", type=int, default=10000, help="number of tokens")
    runner(parser.parse_args())