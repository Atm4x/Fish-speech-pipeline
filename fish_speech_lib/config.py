FIREFLY_GAN_VQ_CONFIG = {
    "_target_": "fish_speech.models.vqgan.modules.firefly.FireflyArchitecture",
    "spec_transform": {
        "_target_": "fish_speech.utils.spectrogram.LogMelSpectrogram",
        "sample_rate": 44100,
        "n_mels": 160,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 2048,
    },
    "backbone": {
        "_target_": "fish_speech.models.vqgan.modules.firefly.ConvNeXtEncoder",
        "input_channels": 160,
        "depths": [3, 3, 9, 3],
        "dims": [128, 256, 384, 512],
        "drop_path_rate": 0.2,
        "kernel_size": 7,
    },
    "head": {
        "_target_": "fish_speech.models.vqgan.modules.firefly.HiFiGANGenerator",
        "hop_length": 512,
        "upsample_rates": [8, 8, 2, 2, 2],  # aka. strides
        "upsample_kernel_sizes": [16, 16, 4, 4, 4],
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "num_mels": 512,
        "upsample_initial_channel": 512,
        "pre_conv_kernel_size": 13,
        "post_conv_kernel_size": 13,
    },
    "quantizer": {
        "_target_": "fish_speech.models.vqgan.modules.fsq.DownsampleFiniteScalarQuantize",
        "input_dim": 512,
        "n_groups": 8,
        "n_codebooks": 1,
        "levels": [8, 5, 5, 5],
        "downsample_factor": [2, 2],
    },
}