def count_parameters_conv(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool
):
    val = (in_channels * kernel_size**2) * out_channels
    if bias:
        val += out_channels
    return val
