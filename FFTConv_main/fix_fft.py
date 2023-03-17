'''
本研究のメインのファイル
カーネルのFFTを外した処理をここで行っている
'''



from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torch.fft import irfftn, rfftn


def complex_matmul(a: Tensor, b: Tensor, groups: int = 1) -> Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).

    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    #dimの調整
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])
    
    # print('signal_view : ' + str(a.size()))
    # print('kernel_view : ' + str(b.size()))

    #movedimでdimの移動,unsqueezeは負の値が入ることで、最後の次元に1次元を挿入する
    a = torch.movedim(a, 2, a.dim() - 1).unsqueeze(-2)
    b = torch.movedim(b, (1, 2), (b.dim() - 1, b.dim() - 2))
    
    # complex value matrix multiplication
    real = a.real @ b.real - a.imag @ b.imag
    imag = a.imag @ b.real + a.real @ b.imag
    
    # print('real : ' + str(real.size()))
    # print('imag : ' + str(imag.size()))
    
    real = torch.movedim(real, real.dim() - 1, 2).squeeze(-1)
    imag = torch.movedim(imag, imag.dim() - 1, 2).squeeze(-1)
    
    # print('real_movedim : ' + str(real.size()))
    # print('imag_movedim : ' + str(imag.size()))
    
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag
    
    # print('c : ' + str(c.size()))
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """
    def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    の -> Tuple[int, ...]:は->により戻り値を指定している。c言語みたいに。
    
    """
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)
    
    
def to_ntuple_kernel(val: Union[int, Iterable[int]], n: int) -> Tensor:
    """
        kernel用
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        # return n * (val,)
        # return (2 * (val - 1), val)
        # return torch.tensor([2 * (val - 1), val] , dtype=torch.int64 ,requires_grad=False)
        return torch.tensor(n * [val,] , dtype=torch.int64 ,requires_grad=False)



def fft_conv(
    signal: Tensor,
    kernel: Tensor,
    kernelsize: Tensor,
    bias: Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    padding_mode: str = "constant",
    stride: Union[int, Iterable[int]] = 1,
    dilation: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).

    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.

    Returns:
        (Tensor) Convolved tensor
    """
    # Cast padding, stride & dilation to tuples.
    n = signal.ndim - 2
    padding_ = to_ntuple(padding, n=n)
    stride_ = to_ntuple(stride, n=n)
    dilation_ = to_ntuple(dilation, n=n) #拡大、今回指定していないから(1,1)が出力される

    # # internal dilation offsets
    # offset = torch.zeros(1, 1, *dilation_, device=signal.device, dtype=signal.dtype)
    # # print('offset : '+ str(offset.size())) #sizeは(1,1,1,1) 値は0
    # offset[(slice(None), slice(None), *((0,) * n))] = 1.0
    # # print('offset : '+ str(offset.size())) #sizeは(1,1,1,1) 値は1


    # #dilation_を指定していないからcutoffには何も入らない。cutoff : (slice(None, None, None), slice(None, None, None))
    # # correct the kernel by cutting off unwanted dilation trailing zeros
    # cutoff = tuple(slice(None, -d + 1 if d != 1 else None) for d in dilation_)
    # # print('cutoff : '+ str(cutoff))

    # # cutoffがNoneで、offsetが1だから、ここの処理ではkernelのsizeが変わっていない -> dilationが設定された時の処理だろう
    # # pad the kernel internally according to the dilation parameters
    # kernel = torch.kron(kernel, offset)[(slice(None), slice(None)) + cutoff]


    #今回paddingは設定していないからsignalは変更されない。
    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)] #padding_[::-1]で-1によりリストの要素を逆順で取り出せる
    signal = f.pad(signal, signal_padding, mode=padding_mode)
    '''
    f.pad(signal, signal_padding, mode=padding_mode)
    例:
    signal : source ----- ([10,11,12,13])
    signal_padding : padding_size ------ ([1,2,3,4])
    mode : 毎回'constant' で設定されている
    padding([1,2,3,4])を((1,2),(3,4))と考えることができる。(1,2)はsignal(3)の13の左一列に0埋め、右二列に0埋めする。よって1行のデータ数が1+13+2=16となる
    (3,4)はsignal(2)の12の上3行を0埋め、下4行を0埋める。よって1列のデータ数3+12+4=19となる
    よってpadding後の出力には([10,11,19,16])となる。
    
    仮に
    signal_padding : padding_size ------ ([1,2,3,4,5,6])
    とすると、(5,6)はsignal(1)の上下にpaddingされる。そのため5+11+16=32。総出力が([10,32,19,16])
    '''
    # print('signal : '+ str(signal.size()))

    #入力画像のrfftでサイズが2分の1になる部分が奇数だったらpaddingして偶数にしている。
    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0: #.size(-1)で一番後ろの要素を取り出している
        signal_ = f.pad(signal, [0, 1])
    else:
        signal_ = signal

    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))

    output_fr = complex_matmul(signal_fr, kernel, groups=groups)

    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # 入力画像から指定したカーネルのサイズを引いたサイズになるようにトリミングする。
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernelsize[i - 2] + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output


class _FFTConv(nn.Module):
    """Base class for PyTorch FFT convolution layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        padding: Union[int, Iterable[int]] = 0,
        padding_mode: str = "constant",
        stride: Union[int, Iterable[int]] = 1,
        dilation: Union[int, Iterable[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        ndim: int = 1,
        select: int = 1,
        args = None,
    ):
        """
        memo:
             Union : 「str or int」のように「いくつかのある型のうちいずれか」にマッチすればOK
             Iterable :  for 文で繰り返せる(使える)オブジェクトまたはそのクラス
            
            ndim: 配列の次元数
        """
        """
        Args:
            in_channels: (int) Number of channels in input tensors
            out_channels: (int) Number of channels in output tensors
            kernel_size: (Union[int, Iterable[int]) Square radius of the kernel
            padding: (Union[int, Iterable[int]) Number of zero samples to pad the
                input on the last dimension.
            stride: (Union[int, Iterable[int]) Stride size for computing output values.
            bias: (bool) If True, includes bias, which is added after convolution
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.select = select

        if in_channels % groups != 0:
            raise ValueError(
                "'in_channels' must be divisible by 'groups'."
                f"Found: in_channels={in_channels}, groups={groups}."
            )
        if out_channels % groups != 0:
            raise ValueError(
                "'out_channels' must be divisible by 'groups'."
                f"Found: out_channels={out_channels}, groups={groups}."
            )

            
        # print('--------------------------------------------')

        self.kernel_size = to_ntuple_kernel(kernel_size, ndim) #ここで次元によってkernelの枚数を増やしてる。2次元なら○x○
        
        if args.size % 2 != 0:
            re_size = args.size + 1
        else:
            re_size = args.size
        if self.select == 1:
            weight = torch.randn(out_channels, in_channels // groups, args.size, (re_size // 2 + 1), dtype=torch.cfloat)
        
        in_size2 = args.size - args.kernel_l1 + 1
        if in_size2 % 2 != 0:
            re_size = in_size2 + 1
        else:
            re_size = in_size2
        if self.select == 2:
            weight = torch.randn(out_channels, in_channels // groups, in_size2, (re_size // 2 + 1), dtype=torch.cfloat)
        
        in_size3 = in_size2 - args.kernel_l2 + 1
        if in_size3 % 2 != 0:
            re_size = in_size3 + 1
        else:
            re_size = in_size3
        if self.select == 3:
            weight = torch.randn(out_channels, in_channels // groups, in_size3, (re_size // 2 + 1), dtype=torch.cfloat)
        
        
        self.weight = nn.Parameter(weight)# parameterという役割をもたせただけ。これによりoptimizerに model.parameters()と書くだけで情報を渡せる
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

        # print('--------------------------------------------')
    def forward(self, signal):
        return fft_conv(
            signal,
            self.weight,
            bias=self.bias,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            kernelsize=self.kernel_size,
        )


FFTConv1d = partial(_FFTConv, ndim=1)
FFTConv2d = partial(_FFTConv, ndim=2)
FFTConv3d = partial(_FFTConv, ndim=3)
