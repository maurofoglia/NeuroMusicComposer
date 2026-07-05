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

    # ==========================================
    # 🧠 FOUNDATION MODELS (Transformer/Modern)
    # ==========================================

    if model_name == "REVE":
        # TRANSFORMER PARAMETERS (Default: REVE-Base)
        extra_kwargs["embed_dim"] = 512
        extra_kwargs["depth"] = 22
        extra_kwargs["heads"] = 8
        extra_kwargs["head_dim"] = 64
        extra_kwargs["mlp_dim_ratio"] = 2.66
        extra_kwargs["use_geglu"] = True

        # POSITIONAL PARAMETERS (Fourier 4D Positional Encoding)
        extra_kwargs["freqs"] = 4

        # PATCHING PARAMETERS (Time windows)
        extra_kwargs["patch_size"] = 256
        extra_kwargs["patch_overlap"] = 20

        # FINAL POOLING STRATEGY
        extra_kwargs["attention_pooling"] = False

    elif model_name in ["BIOT", "InterpolatedBIOT"]:
        # The famous mathematical detail: 128 is 0.5s at 256Hz
        extra_kwargs["hop_length"] = 128
        extra_kwargs["embed_dim"] = 256
        extra_kwargs["num_layers"] = 4
        extra_kwargs["num_heads"] = 8
        extra_kwargs["activation"] = nn.ELU
        extra_kwargs["drop_prob"] = 0.5
        extra_kwargs["att_drop_prob"] = 0.2
        extra_kwargs["att_layer_drop_prob"] = 0.2

    elif model_name in ["Labram", "InterpolatedLaBraM", "InterpolatedLabram"]:
        extra_kwargs["neural_tokenizer"] = True
        extra_kwargs["patch_size"] = 256
        extra_kwargs["learned_patcher"] = False
        extra_kwargs["conv_in_channels"] = 1
        extra_kwargs["conv_out_channels"] = 8
        extra_kwargs["embed_dim"] = 200
        extra_kwargs["num_layers"] = 12
        extra_kwargs["num_heads"] = 10
        extra_kwargs["mlp_ratio"] = 4.0
        extra_kwargs["attn_head_dim"] = None
        extra_kwargs["activation"] = nn.GELU
        extra_kwargs["norm_layer"] = nn.LayerNorm
        extra_kwargs["qk_norm"] = nn.LayerNorm
        extra_kwargs["qkv_bias"] = False
        extra_kwargs["qk_scale"] = None
        extra_kwargs["init_values"] = 0.1
        extra_kwargs["init_scale"] = 0.001
        extra_kwargs["use_abs_pos_emb"] = True
        extra_kwargs["use_mean_pooling"] = False
        extra_kwargs["drop_prob"] = 0.0
        extra_kwargs["attn_drop_prob"] = 0.0
        extra_kwargs["drop_path_prob"] = 0.0

    elif model_name in ["BENDR", "InterpolatedBENDR"]:
        # FEATURE EXTRACTOR PARAMETERS (1D CNN)
        extra_kwargs["encoder_h"] = 512
        extra_kwargs["enc_width"] = (3, 2, 2, 2, 2, 2)
        extra_kwargs["enc_downsample"] = (3, 2, 2, 2, 2, 2)
        extra_kwargs["activation"] = nn.GELU

        # CONTEXTUALIZER PARAMETERS (Transformer)
        extra_kwargs["transformer_layers"] = 8
        extra_kwargs["transformer_heads"] = 8
        extra_kwargs["contextualizer_hidden"] = 3076
        extra_kwargs["position_encoder_length"] = 25
        extra_kwargs["start_token"] = -5

        # REGULARIZATION (0.15 pre-train, 0.0 fine-tune)
        extra_kwargs["drop_prob"] = 0.1
        extra_kwargs["layer_drop"] = 0.0

        # STRUCTURAL MODES
        extra_kwargs["projection_head"] = False
        extra_kwargs["final_layer"] = True
        extra_kwargs["encoder_only"] = False  # True bypasses transformer if validation plateaus

    elif model_name == "EEGPT":
        extra_kwargs["embed_dim"] = 512
        extra_kwargs["depth"] = 8
        extra_kwargs["num_heads"] = 8
        extra_kwargs["mlp_ratio"] = 4.0
        extra_kwargs["patch_size"] = 64
        extra_kwargs["patch_stride"] = 32
        extra_kwargs["embed_num"] = 4
        extra_kwargs["drop_prob"] = 0.0
        extra_kwargs["attn_drop_rate"] = 0.0
        extra_kwargs["drop_path_rate"] = 0.0
        extra_kwargs["chan_proj_type"] = 'conv1d_constraint'
        extra_kwargs["n_chans_target"] = 19
        extra_kwargs["chan_conv_max_norm"] = 1.0
        extra_kwargs["qkv_bias"] = True
        extra_kwargs["init_std"] = 0.02
        extra_kwargs["layer_norm_eps"] = 1e-06
        extra_kwargs["return_encoder_output"] = False

    elif model_name == "LUNA":
        # Base variant defaults (Change to 96/6/10 for Large, 128/8/24 for Huge)
        extra_kwargs["embed_dim"] = 64
        extra_kwargs["num_queries"] = 4
        extra_kwargs["depth"] = 8
        extra_kwargs["num_heads"] = 2
        extra_kwargs["mlp_ratio"] = 4.0
        extra_kwargs["patch_size"] = 40
        extra_kwargs["drop_path"] = 0.0
        extra_kwargs["drop_prob_chan"] = 0.0
        extra_kwargs["attn_drop"] = 0.0
        extra_kwargs["norm_layer"] = nn.LayerNorm
        extra_kwargs["activation"] = nn.GELU

    elif model_name in ["SignalJEPA", "InterpolatedSignalJEPA"]:
        extra_kwargs["feature_encoder__mode"] = "layer_norm"
        extra_kwargs["feature_encoder__conv_bias"] = True
        extra_kwargs["activation"] = nn.GELU

        # ✅ ALLINEATI AL CHECKPOINT UFFICIALE HUGGINGFACE
        extra_kwargs["pos_encoder__spat_dim"] = 30  # L'originale usava 30 (divisibile per 3 e 2!)
        extra_kwargs["pos_encoder__time_dim"] = 34  # (34 + 30 = 64, che è la dimensione del Transformer)

        extra_kwargs["transformer__d_model"] = 64  # L'originale usava 64, non 256
        extra_kwargs["transformer__num_encoder_layers"] = 6
        extra_kwargs["transformer__num_decoder_layers"] = 0
        extra_kwargs["transformer__nhead"] = 8
        extra_kwargs["drop_prob"] = 0.1
        extra_kwargs["channel_embedding"] = "scratch"

    elif model_name == "CBraMod":
        extra_kwargs["emb_dim"] = 200
        extra_kwargs["n_layer"] = 12
        extra_kwargs["nhead"] = 8
        extra_kwargs["dim_feedforward"] = 800
        extra_kwargs["activation"] = nn.GELU
        extra_kwargs["patch_size"] = 200  # Defaults to 1s at 200Hz
        extra_kwargs["channels_kernel_stride_padding_norm"] = (
            (25, 49, 25, 24, (5, 25)),
            (25, 3, 1, 1, (5, 25)),
            (25, 3, 1, 1, (5, 25))
        )
        extra_kwargs["drop_prob"] = 0.1
        extra_kwargs["return_encoder_output"] = False


    # ==========================================
    # ⚙️ CLASSIC MODELS (Braindecode Standard)
    # ==========================================

    elif model_name == "CTNet":
        extra_kwargs["chs_info"] = chs_info
        extra_kwargs["activation_patch"] = nn.ELU
        extra_kwargs["activation_transformer"] = nn.GELU
        extra_kwargs["n_filters_time"] = 20
        extra_kwargs["kernel_size"] = 64
        extra_kwargs["depth_multiplier"] = 2
        extra_kwargs["pool_size_1"] = 8
        extra_kwargs["pool_size_2"] = 8
        extra_kwargs["num_heads"] = 4
        extra_kwargs["embed_dim"] = 40
        extra_kwargs["num_layers"] = 6
        extra_kwargs["cnn_drop_prob"] = 0.3
        extra_kwargs["att_positional_drop_prob"] = 0.1
        extra_kwargs["final_drop_prob"] = 0.5

    elif model_name == "EEGNetv4":
        extra_kwargs["chs_info"] = chs_info
        extra_kwargs["final_conv_length"] = 'auto'
        extra_kwargs["kernel_length"] = 64
        extra_kwargs["depthwise_kernel_length"] = 16
        extra_kwargs["F1"] = 8
        extra_kwargs["D"] = 2
        extra_kwargs["F2"] = None  # Auto-calculates as F1 * D
        extra_kwargs["pool_mode"] = 'mean'
        extra_kwargs["pool1_kernel_size"] = 4
        extra_kwargs["pool2_kernel_size"] = 8
        extra_kwargs["drop_prob"] = 0.25
        extra_kwargs["conv_spatial_max_norm"] = 1.0
        extra_kwargs["batch_norm_momentum"] = 0.01
        extra_kwargs["batch_norm_affine"] = True
        extra_kwargs["batch_norm_eps"] = 1e-3
        extra_kwargs["activation"] = nn.ELU

    return extra_kwargs