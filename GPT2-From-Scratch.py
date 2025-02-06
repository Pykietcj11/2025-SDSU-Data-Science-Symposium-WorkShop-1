import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        """
        Initialize the dataset by tokenizing text and creating input-output chunks.

        Args:
            txt (str): The input text to be tokenized and split into training chunks.
            tokenizer: A tokenizer that converts text into token IDs.
            max_length (int): The maximum length of each tokenized chunk.
            stride (int): The step size for moving through the token sequence.
        """
        self.input_ids = []  # List to store input token sequences
        self.target_ids = []  # List to store target token sequences

        # Step 1: Tokenize the entire input text into token IDs
        token_ids = tokenizer.encode(txt)  # Convert text to a list of token IDs

        # Step 2: Create overlapping tokenized chunks for training
        # We slide over the text using 'stride' to create multiple overlapping samples
        for i in range(0, len(token_ids) - max_length, stride):
            # Get the input chunk of max_length tokens
            input_chunk = token_ids[i:i + max_length]

            # The target chunk is the same as the input but shifted one step forward
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Convert token lists to PyTorch tensors and store them
            self.input_ids.append(torch.tensor(input_chunk))  
            self.target_ids.append(torch.tensor(target_chunk))  

    def __len__(self):
        """
        Returns:
            int: The total number of training samples (input-output pairs).
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Fetch a specific training example.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input token tensor, target token tensor)
        """
        return self.input_ids[idx], self.target_ids[idx]




def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """
    Create a DataLoader for tokenized text sequences using GPTDatasetV1.

    Args:
        txt (str): The input text to be processed into tokenized chunks.
        batch_size (int, optional): Number of samples per batch. Default is 4.
        max_length (int, optional): Maximum sequence length for tokenized text. Default is 256.
        stride (int, optional): Step size for overlapping chunks. Default is 128.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.
        drop_last (bool, optional): Whether to drop the last batch if it's smaller than batch_size. Default is True.
        num_workers (int, optional): Number of subprocesses for data loading. Default is 0 (main process).

    Returns:
        DataLoader: A PyTorch DataLoader object for batched training.
    """
    
    # Step 1: Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")  # Using GPT-2 encoding for tokenization

    # Step 2: Create dataset using the provided text and tokenizer
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Step 3: Initialize DataLoader with dataset and batching configurations
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples per batch
        shuffle=shuffle,        # Shuffle data during training for randomness
        drop_last=drop_last,    # If True, removes the last batch if it's incomplete
        num_workers=num_workers # Number of worker threads for parallel data loading
    )

    return dataloader  # Return the DataLoader object for use in training or inference




def generate_embeddings(dataloader, token_embedding_layer, output_dim):
    """
    Processes data from the DataLoader and generates token + positional embeddings.

    Args:
        dataloader (DataLoader): The DataLoader object created using create_dataloader_v1.
        token_embedding_layer (nn.Embedding): The embedding layer for token embeddings.
        output_dim (int): The dimension of the embeddings.

    Returns:
        tuple: (input_embeddings, token_embeddings, pos_embeddings)
            - input_embeddings (Tensor): Combined token + position embeddings.
            - token_embeddings (Tensor): Embeddings generated for tokens.
            - pos_embeddings (Tensor): Positional embeddings.
    """
    
    # Step 1: Get the first batch from DataLoader
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)  # Retrieve the first batch

    # Step 2: Generate token embeddings
    token_embeddings = token_embedding_layer(inputs)

    # Step 3: Create positional embeddings
    context_length = inputs.shape[1]  # Max sequence length from batch
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    # Step 4: Combine token embeddings with positional embeddings
    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings, token_embeddings, pos_embeddings



class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        
        # Ensure d_out is divisible by num_heads (each head gets equal dimensions)
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Each head's dimensionality
        
        # Linear layers to project input into Q, K, V
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
        # Final projection layer after attention computation
        self.out_proj = nn.Linear(d_out, d_out)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask to prevent attending to future tokens (for autoregressive models)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Compute Query, Key, and Value projections
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)       # (b, num_tokens, d_out)
        values = self.W_value(x)   # (b, num_tokens, d_out)
        
        # Reshape to separate attention heads
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Rearrange to shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Compute attention scores using scaled dot-product
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # (b, num_heads, num_tokens, num_tokens)
        
        # Apply causal mask (if applicable) to prevent attending to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        
        # Compute attention weights using softmax (with scaling for stability)
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)  # Apply dropout for regularization
        
        # Compute weighted sum of values (attention output)
        context_vec = torch.matmul(attn_weights, values)  # (b, num_heads, num_tokens, head_dim)
        
        # Restore original dimensions: (b, num_tokens, d_out)
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        
        # Apply final output projection
        context_vec = self.out_proj(context_vec)
        
        return context_vec


        class FeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # Defines a two-layer feed-forward network with GELU activation
        self.layers = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # Expands dimension by a factor of 4
            nn.GELU(),  # Applies GELU activation
            nn.Linear(4 * d_model, d_model),  # Compresses back to original size
            nn.Dropout(dropout)  # Adds dropout for regularization
        )
    
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, context_length, dropout=0.1):
        super().__init__()
        # Multi-head self-attention mechanism
        self.att = MultiHeadAttention(d_model, d_model, context_length, dropout, num_heads)
        # Feed-forward network
        self.ff = FeedForward(d_model, dropout)
        # Layer normalization to stabilize training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout for regularization
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # First shortcut connection and layer normalization before attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        x = x + shortcut  # Residual connection
        
        # Second shortcut connection and layer normalization before feed-forward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut  # Residual connection
        
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()
        # Token embedding layer: converts token indices into dense vectors
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        # Positional embedding layer: learns positional information of tokens
        self.pos_emb = nn.Embedding(context_length, d_model)
        # Dropout applied after embedding
        self.drop_emb = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, context_length, dropout) for _ in range(num_layers)]
        )
        
        # Final layer normalization before output
        self.final_norm = nn.LayerNorm(d_model)
        # Output layer projecting to vocabulary size for token prediction
        self.out_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # Get token embeddings
        tok_embeds = self.tok_emb(in_idx)
        # Get positional embeddings
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        # Combine token and positional embeddings
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)  # Apply dropout
        
        # Pass through transformer blocks
        x = self.trf_blocks(x)
        
        # Apply final normalization and output projection
        x = self.final_norm(x)
        logits = self.out_head(x)  # Compute logits for vocabulary
        
        return logits
