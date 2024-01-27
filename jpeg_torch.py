import itertools
import torch
import numpy as np



def rgb_to_ycbcr_jpeg_torch(image):
    matrix = torch.tensor(
        [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=torch.float32).T

    shift = torch.tensor([0., 128., 128.], dtype=torch.float32)
    result = torch.tensordot(image, matrix, dims=1) + shift
    result = result.view(image.size())
    return result


# 2. Chroma subsampling
def downsampling_420_torch(image):
    # input: batch x height x width x 3
    # output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    y, cb, cr = torch.split(image.permute(0,3,1,2), 1, dim=1)
    
    print("y:{} cb:{} cr:{}".format(y.shape,cb.shape,cr.shape))
    cb = torch.nn.functional.avg_pool2d(cb, kernel_size=2, stride=2, padding=0)
    cr = torch.nn.functional.avg_pool2d(cr, kernel_size=2, stride=2, padding=0)
    return (y.squeeze(dim=1), cb.squeeze(dim=1), cr.squeeze(dim=1))

def dct_8x8_torch(image):
    image = image - 128
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    
    tensor = torch.tensor(tensor, dtype=torch.float32)
    scale = torch.tensor(scale, dtype=torch.float32)
    
    result = scale * torch.tensordot(image, tensor, dims=2)
    result = result.view(image.size())
    return result

# 3. Block splitting
def image_to_patches_torch(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    k = 8
    height, width = image.size()[1:3]
    batch_size = image.size(0)
    image_reshaped = image.reshape(batch_size, height // k, k, -1, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.reshape(batch_size, -1, k, k)

def idct_8x8_torch(image):
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    image = image * alpha

    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    print("image:",type(image))
    tensor = torch.tensor(tensor, dtype=image.dtype)

    result = 0.25 * torch.tensordot(image, tensor, dims=2) + 128
    result = result.view(image.size())
    return result

# -3. Block joining
def patches_to_image_torch(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    height, width = int(height), int(width)
    k = 8
    batch_size = patches.size(0)
    image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    return image_transposed.reshape(batch_size, height, width)


def upsampling_420_torch(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x height x width x 3

    def repeat(x, k=2):
        batch_size, height, width = x.size()
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, k, k)
        x = x.view(-1, height * k, width * k)
        return x

    cb = repeat(cb)
    cr = repeat(cr)

    image = torch.stack((y, cb, cr), dim=-1)
    return image


def ycbcr_to_rgb_jpeg_torch(image):
    matrix = torch.tensor(
        [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
        dtype=image.dtype).T
    shift = torch.tensor([0, -128, -128], dtype=image.dtype)

    result = torch.tensordot(image + shift, matrix, dims=1)
    result = result.reshape(image.size())
    return result


# 5. Quantizaztion
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T

y_table = torch.tensor(y_table, dtype=torch.float32)
c_table = torch.tensor(c_table, dtype=torch.float32)

def y_quantize_torch(image, rounding=torch.round, factor=1):
  image = image / (y_table * factor)
  image = rounding(image)
  return image


def c_quantize_torch(image, rounding=torch.round, factor=1):
  image = image / (c_table * factor)
  image = rounding(image)
  return image

# -5. Dequantization
def y_dequantize_torch(image, factor=1):
  return image * (y_table * factor)

def c_dequantize_torch(image, factor=1):
  return image * (c_table * factor)


def jpeg_compress_decompress_torch(image, downsample_c=True, rounding=torch.round, factor=1):
    image = torch.tensor(image, dtype=torch.float32)
    height, width = image.size()[1:3]
    orig_height, orig_width = height, width

    # Compression
    image = rgb_to_ycbcr_jpeg_torch(image)

    if downsample_c:
        y, cb, cr = downsampling_420_torch(image)
    else:
        y, cb, cr = torch.split(image.permute(0,3,1,2), 1, dim=1)
    components = {'y': y, 'cb': cb, 'cr': cr}

    for k in components.keys():
        comp = components[k]
        comp = image_to_patches_torch(comp)
        comp = dct_8x8_torch(comp)

        comp = c_quantize_torch(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize_torch(comp, rounding, factor)
        components[k] = comp

    # Decompression
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize_torch(comp, factor) if k in ('cb', 'cr') else y_dequantize_torch(comp, factor)
        comp = idct_8x8_torch(comp)
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image_torch(comp, height // 2, width // 2)
            else:
                comp = patches_to_image_torch(comp, height, width)
        else:
            comp = patches_to_image_torch(comp, height, width)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420_torch(y, cb, cr)
    else:
        image = torch.stack((y, cb, cr), dim=-1)
    image = ycbcr_to_rgb_jpeg_torch(image)

    # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
    # min_value = torch.minimum(torch.min(image), 0.)
    # max_value = torch.maximum(torch.max(image), 255.)
    # value_range = max_value - min_value
    # image = 255 * (image - min_value) / value_range
    image = torch.minimum(torch.tensor(255.), torch.maximum(torch.tensor(0.), image))
    return image




if __name__ == '__main__':
  import cv2
  image = np.expand_dims(cv2.imread('images/goldfish.jpg'),axis=0)
  #image = np.random.rand(1,512,512) #DCT
  #image = np.random.rand(1,4096,8,8) #IDCT

  #tf_image = tf.convert_to_tensor(image,tf.float32)
  torch_image = torch.from_numpy(image).to(dtype=torch.float32)
  reconstructed_img = jpeg_compress_decompress_torch(torch_image, downsample_c=True,rounding=torch.round,factor=1)
  print("reconstructed image:",reconstructed_img)
  print("image:{} reconstructed_image:{}".format(torch_image.shape,reconstructed_img.shape)) 
  cv2.imwrite('jpeg_torch_recon.png',reconstructed_img.numpy().squeeze())

  '''
  #print("input image:",image.shape)
  #tf_result = rgb_to_ycbcr_jpeg(tf_image)
  #torch_result = rgb_to_ycbcr_jpeg_torch(torch_image)
  
  y_tf, cb_tf, cr_tf = downsampling_420(tf_image)
  y_torch, cb_torch, cr_torch = downsampling_420_torch(torch_image)
  
  y_torch_patches = image_to_patches_torch(y_torch)
  y_torch_quantized_c = c_quantize_torch(y_torch_patches)
  y_torch_dequantized_c = c_dequantize_torch(y_torch_quantized_c)

  y_torch_quantized_y = y_quantize_torch(y_torch_patches)
  y_torch_dequantized_y = y_dequantize_torch(y_torch_quantized_y)

  print("y_torch_dequantized(c):{}   y_torch_dequantized(y):{}".format(y_torch_dequantized_c,y_torch_dequantized_y))


  #tf_result = image_to_patches(tf_image)
  #torch_result = image_to_patches_torch(torch_image)
  #tf_result = idct_8x8(tf_image)
  #torch_result = idct_8x8_torch(torch_image)
  #tf_result = patches_to_image(tf_image,height=512,width=512)
  #torch_result = patches_to_image_torch(torch_image,height=512,width=512)
  #tf_result = upsampling_420(y_tf, cb_tf, cr_tf)
  #torch_result = upsampling_420_torch(y_torch, cb_torch, cr_torch)

  #tf_result = ycbcr_to_rgb_jpeg(tf_result)
  #torch_result = ycbcr_to_rgb_jpeg_torch(torch_result)

  #tf_result = tf_result.numpy()
  #torch_result = torch_result.numpy()

  #print("tf_result:{}\n\n torch_result:{}\n\n".format(tf_result,torch_result))
  #print("shape: tf-{}  torch-{}".format(tf_result.shape,torch_result.shape))
  '''


