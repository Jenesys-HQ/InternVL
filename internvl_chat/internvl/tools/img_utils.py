import logging
import requests

import torch
import base64
import io
from io import BytesIO
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Model Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values).to('cuda', dtype=torch.bfloat16)
    return pixel_values


def load_image_bs64(image_bs64, input_size=448, max_num=6):
    image_file = base64.b64decode(image_bs64)
    image_file = BytesIO(image_file)
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_pdf_base64_from_img_url(pdf_url):
    """
    Request a PDF file from a URL and convert it to base64.
    """
    try:
        # Request the PDF from the URL
        response = requests.get(pdf_url)
        response.raise_for_status()  # Ensure the request was successful
        pdf_bytes = response.content

        # Encode the PDF file to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        # Debug: Check the first few bytes of the PDF data
        logger.debug(f"First 5 bs64 of the PDF data: {pdf_base64[:5]}")

        return pdf_base64
    except Exception as e:
        logger.error(e)
        return None


def pdfs_to_images_base64_function(pdf_base64_strings, page_number=0):
    """
    Convert a list of PDF base64 strings to image base64 strings.

    Args:
        pdf_base64_strings (list): The list of PDF base64 strings to be converted.
        page_number (int): The page number to convert, starting from 0.

    Returns:
        list: The list of image base64 strings.
    """
    image_base64_strings = []

    for pdf_base64_string in pdf_base64_strings:
        # decode the PDF base64 string to bytes
        pdf_bytes = base64.b64decode(pdf_base64_string)

        # convert the PDF bytes to a list using PDF2image
        images = convert_from_bytes(pdf_bytes)

        # choose the page to convert to an image
        image = images[page_number]
        # convert the pdf2image format to a base64 string
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_base64_strings.append(base64.b64encode(buffer.getvalue()).decode())

    return image_base64_strings


def pdf_to_image_base64_function(pdf_base64_string, page_number=0):

    # decode the PDF base64 string to bytes
    pdf_bytes = base64.b64decode(pdf_base64_string)

    # convert the PDF bytes to a list using PDF2image
    images = convert_from_bytes(pdf_bytes)

    # choose the page to convert to an image
    image = images[page_number]
    # convert the pdf2image format to a base64 string
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()

