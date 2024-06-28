defmodule AxonOnnx.GridSample do
  import Nx.Defn

  # Main function to perform grid sampling
  defn grid_sample(input, grid, opts \\ []) do
    opts = Keyword.put_new(opts, :align_corners, false)
    opts = Keyword.put_new(opts, :padding_mode, :zeros)  # :zeros, :border, :reflection

    {n, c, h, w} = Nx.shape(input)
    {_, _, out_h, out_w, _} = Nx.shape(grid)

    output = Nx.iota({n, c, out_h, out_w}, type: {:f, 32})

    Nx.Defn.map(output, fn _, idx ->
      batch_idx = elem(idx, 0)
      channel_idx = elem(idx, 1)
      out_y = elem(idx, 2)
      out_x = elem(idx, 3)

      grid_x = grid[{batch_idx, out_y, out_x, 0}]
      grid_y = grid[{batch_idx, out_y, out_x, 1}]

      x = denormalize(grid_x, w, opts[:align_corners])
      y = denormalize(grid_y, h, opts[:align_corners])

      interpolated_value = bilinear_interpolate(input[{batch_idx, channel_idx}], x, y, w, h, opts[:padding_mode])
      Nx.reshape(interpolated_value, {})
    end)
  end

  # Denormalize function to map normalized grid coordinates to actual pixel indices
  defn denormalize(coord, length, align_corners) do
    if align_corners do
      (coord + 1.0) * (length - 1) / 2.0
    else
      ((coord + 1.0) * length - 1) / 2.0
    end
  end

  # Bilinear interpolation
  defn bilinear_interpolate(input, x, y, w, h, padding_mode) do
    x0 = Nx.floor(x)
    x1 = x0 + 1
    y0 = Nx.floor(y)
    y1 = y0 + 1

    # Get values at corner points
    q11 = get_pixel(input, x0, y0, w, h, padding_mode)
    q21 = get_pixel(input, x1, y0, w, h, padding_mode)
    q12 = get_pixel(input, x0, y1, w, h, padding_mode)
    q22 = get_pixel(input, x1, y1, w, h, padding_mode)

    # Interpolation weights
    dx1 = x - x0
    dx2 = 1.0 - dx1
    dy1 = y - y0
    dy2 = 1.0 - dy1

    # Bilinear interpolation formula
    interp_val = q11 * dx2 * dy2 +
                 q21 * dx1 * dy2 +
                 q12 * dx2 * dy1 +
                 q22 * dx1 * dy1
    interp_val
  end

  # Function to get pixel value considering padding modes
  defn get_pixel(input, x, y, w, h, padding_mode) do
    cond do
      x < 0 or x >= w or y < 0 or y >= h ->
        case padding_mode do
          :zeros -> 0.0
          :border -> Nx.slice(input, [Nx.clip(y, 0, h - 1), Nx.clip(x, 0, w - 1)], [1, 1])
          :reflection -> Nx.slice(input, [reflect(y, 0, h - 1), reflect(x, 0, w - 1)], [1, 1])
        end
      true ->
        Nx.slice(input, [y, x], [1, 1])
    end
  end

  # Reflect coordinates within the image dimensions
  defn reflect(coord, min, max) do
    if coord < min do
      2 * min - coord
    else
      if coord > max do
        2 * max - coord
      else
        coord
      end
    end
  end
end
