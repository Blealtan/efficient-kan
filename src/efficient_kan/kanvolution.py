import torch
import torch.nn.functional as F
import math

class Kanv2d(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features, # conv specific
        kernel_size, # conv specific
        stride=1, # conv specific; NOT implemented yet
        padding = 0, # conv specific
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.Identity,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(Kanv2d, self).__init__()

        # Conv parts
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # The later parts are KAN specific, they should be able to stay
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            # BLOCKER: Needs fixing
            #.expand(kernel_size, -1)
            #.expand(kernel_size, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        #self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order, kernel_size, kernel_size)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        #torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.kernel_size, self.kernel_size, self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1, self.kernel_size, self.kernel_size)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1, :, :]) & (x < grid[:, 1:, :, :])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1), :, :])
                / (grid[:, k:-1, :, :] - grid[:, : -(k + 1), :, :])
                * bases[:, :, :-1, :, :]
            ) + (
                (grid[:, k + 1 :, :, :] - x)
                / (grid[:, k + 1 :, :, :] - grid[:, 1:(-k), :, :])
                * bases[:, :, 1:, :, :]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
            self.kernel_size,
            self.kernel_size,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (self.kernel_size, self.kernel_size, x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order, self.kernel_size, self.kernel_size)
        B = y.transpose(2, 3)  # (self.kernel_size, self.kernel_size, in_features, batch_size, out_features)
        print(A.shape)
        print(B.shape)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (self.kernel_size, self.kernel_size, in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order, self.kernel_size, self.kernel_size)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
            self.kernel_size,
            self.kernel_size,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # https://discuss.pytorch.org/t/creating-a-custom-convolutional-layer/191654/7
    def convolve(self, x):
        batch_size = x.shape[0]
        image_channels = x.shape[1]
        image_height = x.shape[2]
        image_width = x.shape[3]

        out_channels = self.spline_weight.shape[0]
        in_channels = self.spline_weight.shape[1]
        kernel_height = self.spline_weight.shape[-2]
        kernel_width = self.spline_weight.shape[-1]

        assert(image_channels == in_channels)
        assert(kernel_height == kernel_width)
        
        # F.unfold takes an input tensor and extracts sliding local blocks (or patches) from it. 
        # These blocks are the regions of the input tensor over which the convolution operation 
        # (filter application) will take place.
        # x_unfolded, will have a shape of [batch_size, in_channels * kernel_height * kernel_width, num_patches]
        # The output will look something like this:
        # tensor([[[ 1.,  2.,  3.,  5.,  6.,  7.,  9., 10., 11.],
        #          [ 2.,  3.,  4.,  6.,  7.,  8., 10., 11., 12.],
        #          [ 5.,  6.,  7.,  9., 10., 11., 13., 14., 15.],
        #          [ 6.,  7.,  8., 10., 11., 12., 14., 15., 16.]]])
        # The first patch is the first element of each row: [1, 2, 5, 6]
        x_unfolded = F.unfold(x, kernel_size=kernel_height, padding=self.padding)

        unfolded_batch_size = x_unfolded.shape[0]
        unfolded_patch_size = x_unfolded.shape[1]
        num_patches = x_unfolded.shape[2]
        assert(unfolded_batch_size == batch_size)
        assert(unfolded_patch_size == in_channels * (self.grid_size + self.spline_order) * kernel_height * kernel_width)

        # Reshape x_unfolded into a format that aligns with the convolution weights A
        # transpose dimensions 1 and 2 above into [batch, num_patches, in_channels * kernel_height * kernel_width]
        # then view as [batch, num_patches, in_channels, kernel_height, kernel_width]
        x_unfolded = x_unfolded.permute(0, 2, 1).view(batch_size, num_patches, in_channels, self.grid_size + self.spline_order, kernel_height, kernel_width)

        # Expand x_unfolded across output_channels to match the dimensions of B_expanded
        x_expanded = x_unfolded.unsqueeze(2).expand(batch_size, num_patches, out_channels, in_channels, self.grid_size + self.spline_order, kernel_height, kernel_width)
    
        return x_expanded

    
    def forward(self, x):
        batch_size = x.shape[0]
        image_channels = x.shape[1]
        image_height = x.shape[2]
        image_width = x.shape[3]

        out_channels = self.out_features
        in_channels = self.in_features
        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]
        
        assert(image_channels == in_channels)
        assert(kernel_height == kernel_width)
        
        # Perform the convolution operation
        x_convolve = self.convolve(x)
        output = torch.sum(x_convolve, dim=[3, 4, 5])

        # Reshape the output
        # Calculate the dimensions of the output feature map
        output_height = (image_height + 2 * self.padding - (kernel_height - 1) - 1) + 1
        output_width = (image_width + 2 * self.padding - (kernel_width - 1) - 1) + 1

        # Reshape output to the shape (batch_size, out_channels, output_height, output_width)
        output = output.view(batch_size, out_channels, output_height, output_width)

        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KANvolution(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.Identity,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANvolution, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                Kanv2d(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

if __name__ == "__main__":
    # With square kernels and equal stride
    m = Kanv2d(16, 33, 3, stride=1)
    input = torch.randn(20, 16, 25, 50)
    output = m(input)
