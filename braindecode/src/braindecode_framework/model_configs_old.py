import torch.nn as nn


def get_model_extra_kwargs(model_name, braindecode_dataset):
    """
    Centralized manager for model-specific parameters.
    Returns a dictionary with the kwargs necessary for initialization.
    """
    extra_kwargs = {}

    # Extract channel information (taking the first available emotion is sufficient)
    first_emotion = list(braindecode_dataset.keys())[0]
    primo_dataset_eeg = braindecode_dataset[first_emotion].datasets[0]
    chs_info = primo_dataset_eeg.raw.info['chs']

    # MODEL RULES DICTIONARY

    # if model_name in ["InterpolatedBIOT", "REVE", "Labram"]: extra_kwargs["chs_info"] = chs_info

    if model_name == "REVE":
        # SPATIAL INFORMATION (Mandatory for REVE)
        extra_kwargs["chs_info"] = chs_info

        # TRANSFORMER PARAMETERS (Default: REVE-Base)
        extra_kwargs["embed_dim"] = 512  # Embedding dimension (1250 for REVE-Large)
        extra_kwargs["depth"] = 22  # Number of Transformer layers
        extra_kwargs["heads"] = 8  # Number of attention heads
        extra_kwargs["head_dim"] = 64  # Dimension of each individual head
        extra_kwargs["mlp_dim_ratio"] = 2.66  # Multiplier for feed-forward layers (GEGLU)
        extra_kwargs["use_geglu"] = True  # Use GEGLU (recommended) instead of GELU

        # POSITIONAL PARAMETERS (Fourier 4D Positional Encoding)
        extra_kwargs["freqs"] = 4  # Frequencies for Fourier embedding

        # PATCHING PARAMETERS (Time windows)
        # The official default is 200, but we set 256 to fit your 512 window perfectly
        extra_kwargs["patch_size"] = 256
        extra_kwargs["patch_overlap"] = 20  # Overlap between patches

        # FINAL POOLING STRATEGY
        extra_kwargs["attention_pooling"] = False  # If True: uses attention to aggregate. If False: uses Flatten + Linear.

    elif model_name == "BIOT":
        """
        IMPORTANT MATHEMATICAL DETAIL: BIOT documentation indicates hop_length=100 for a 200Hz signal 
        (i.e., it hops by half a second). 
        Since you are passing data at 256Hz, I have preset hop_length=128 
        to maintain the same temporal overlap logic intended by the authors!
        """
        # Here we use extra_kwargs and pass FIXED numbers!
        extra_kwargs["chs_info"] = chs_info

        # The famous mathematical detail
        extra_kwargs["hop_length"] = 128

        extra_kwargs["embed_dim"] = 256
        extra_kwargs["num_layers"] = 4
        extra_kwargs["num_heads"] = 8

        # Disable dropouts if we want fixed parameters
        extra_kwargs["drop_prob"] = 0.5
        extra_kwargs["att_drop_prob"] = 0.2
        extra_kwargs["att_layer_drop_prob"] = 0.2

    elif model_name in ["Labram", "InterpolatedLaBraM", "InterpolatedLabram"]:
        # PLEASE NOTE ON patch_size: preset patch_size to 256 (instead of the default 200)
        # because window (n_times) is 512. If use 200, the model will crash.
        # Can use 64, 128, or 256 (any integer divisor of 512).

        # SPATIAL INFORMATION
        extra_kwargs["chs_info"] = chs_info

        # MODEL MODE
        # True = Neural Tokenizer (used for classification/feature extraction)
        # False = Neural Decoder (used to reconstruct the original signal)
        extra_kwargs["neural_tokenizer"] = True

        # PATCHER PARAMETERS (Signal segmenter)
        extra_kwargs["patch_size"] = 256  # MUST be a divisor of input_window_samples (e.g., 512 -> 256, 128, 64)
        extra_kwargs[
            "learned_patcher"] = False  # False = use fixed rearrangement. True = use Conv1D to create patches

        # CONVOLUTIONAL PARAMETERS (If neural_tokenizer=True)
        extra_kwargs["conv_in_channels"] = 1
        extra_kwargs["conv_out_channels"] = 8

        # TRANSFORMER ARCHITECTURE
        extra_kwargs["embed_dim"] = 200  # Embedding dimension
        extra_kwargs["num_layers"] = 12  # Number of attention layers
        extra_kwargs["num_heads"] = 10  # Number of attention heads (embed_dim must be divisible by num_heads)
        extra_kwargs["mlp_ratio"] = 4.0  # Multiplier for feedforward layers
        extra_kwargs["attn_head_dim"] = None  # Specific head dimension (None = auto-calculated)

        # NORMALIZATION, ACTIVATION AND SCALING
        extra_kwargs["activation"] = nn.GELU  # Activation function
        extra_kwargs["norm_layer"] = nn.LayerNorm  # Main normalization
        extra_kwargs["qk_norm"] = nn.LayerNorm  # Normalization for Query and Key (Set to None to disable)
        extra_kwargs["qkv_bias"] = False  # Adds bias to Query, Key, and Value
        extra_kwargs["qk_scale"] = None  # Scale factor (None = automatic)
        extra_kwargs["init_values"] = 0.1  # Initialization for residual scaling
        extra_kwargs["init_scale"] = 0.001  # Global initial weight scale

        # POOLING AND POSITION STRATEGIES
        extra_kwargs["use_abs_pos_emb"] = True  # Use absolute positional embeddings
        extra_kwargs["use_mean_pooling"] = False  # Use mean pooling for output instead of [CLS] token

        # DROPOUT AND REGULARIZATION (Leave at 0.0 if handled by Optuna)
        extra_kwargs["drop_prob"] = 0.0  # General dropout
        extra_kwargs["attn_drop_prob"] = 0.0  # Specific dropout for attention matrix
        extra_kwargs["drop_path_prob"] = 0.0  # Dropout for residual paths (Stochastic Depth)

    elif model_name == "BENDR":

        # CHANNEL INFO (Even though BENDR uses channel-independent conv1D,
        # spatial info can be included for completeness if required)
        extra_kwargs["chs_info"] = chs_info

        # FEATURE EXTRACTOR PARAMETERS (1D CNN)
        extra_kwargs["encoder_h"] = 512  # BENDR vector dimension (output channels)

        extra_kwargs["enc_width"] = (3, 2, 2, 2, 2, 2)  # Kernel size of the 6 convolutional blocks

        extra_kwargs["enc_downsample"] = (3, 2, 2, 2, 2, 2)  # Stride of the 6 blocks (compresses signal 96x)

        extra_kwargs["activation"] = nn.GELU  # Activation function

        # CONTEXTUALIZER PARAMETERS (Transformer)
        extra_kwargs["transformer_layers"] = 8  # Number of Transformer layers

        extra_kwargs["transformer_heads"] = 8  # Number of attention heads

        extra_kwargs["contextualizer_hidden"] = 3076  # Hidden feedforward dimension (approx 2x dimension)

        extra_kwargs["position_encoder_length"] = 25  # Receptive field of convolutional positional encoding

        extra_kwargs["start_token"] = -5  # Fixed start token value

        # DROPOUT AND REGULARIZATION (Pre-training: 0.15 and 0.01 / Fine-tuning: 0.0)
        extra_kwargs["drop_prob"] = 0.1  # General model dropout

        extra_kwargs["layer_drop"] = 0.0  # Probability of dropping entire Transformer layers

        # STRUCTURAL MODES
        extra_kwargs["projection_head"] = False  # Used only for self-supervised pre-training

        extra_kwargs["final_layer"] = True  # True = Adds the final linear layer for classification

        # THE BENDR MAGIC PARAMETER
        # Documentation says that bypassing the Transformer (encoder_only=True)
        # outperformed the full model in 4 out of 5 tasks if not pre-training from scratch.
        # recommended keeping this on True if the loss isn't decreasing!
        extra_kwargs["encoder_only"] = False

    elif model_name == "CTNet":
        # CHANNEL INFO
        extra_kwargs["chs_info"] = chs_info

        # ACTIVATION FUNCTIONS
        extra_kwargs["activation_patch"] = nn.ELU
        extra_kwargs["activation_transformer"] = nn.GELU

        # CONVOLUTIONAL PARAMETERS (CNN)
        extra_kwargs["n_filters_time"] = 20  # Initial temporal filters
        extra_kwargs["kernel_size"] = 64  # Temporal kernel size
        extra_kwargs["depth_multiplier"] = 2  # Depth-wise filter multiplier
        extra_kwargs["pool_size_1"] = 8  # First pooling
        extra_kwargs["pool_size_2"] = 8  # Second pooling

        # TRANSFORMER PARAMETERS
        extra_kwargs["num_heads"] = 4  # Attention heads
        extra_kwargs["embed_dim"] = 40  # Embedding dimension
        extra_kwargs["num_layers"] = 6  # Number of encoder layers

        # DROPOUTS (Activate here ONLY if not optimizing with Optuna)
        extra_kwargs["cnn_drop_prob"] = 0.3
        extra_kwargs["att_positional_drop_prob"] = 0.1
        extra_kwargs["final_drop_prob"] = 0.5

    elif model_name == "EEGNetv4":
        pass

    return extra_kwargs