import fitz  # PyMuPDF
import cv2
import numpy as np
import concurrent.futures
from matplotlib import pyplot as plt
import os
from PIL import Image
import argparse



def stack_images(img1, img2, img3, img4):
    # Resize images to have the same dimensions (optional)
    # This assumes that all images have the same size
    img1 = cv2.resize(img1, (0,0), fx=0.5, fy=0.5)
    img2 = cv2.resize(img2, (0,0), fx=0.5, fy=0.5)
    img3 = cv2.resize(img3, (0,0), fx=0.5, fy=0.5)
    img4 = cv2.resize(img4, (0,0), fx=0.5, fy=0.5)

    # Stack images in a 2x2 matrix
    top_row = np.hstack((img1, img2))
    bottom_row = np.hstack((img3, img4))
    result_img = np.vstack((top_row, bottom_row))

    return result_img


def resize(image):
    height, width = image.shape[:2]
    max_dimension = max(height, width)
    quotient = max_dimension // 1500
    rem = max_dimension % 1500
    # If the max dimension is less than or equal to 1500, no resizing is needed
    if rem == 0:
      return cv2.resize(image, (max_dimension, max_dimension))

    # Calculate new dimensions
    # The +1 ensures we scale up to the next full 1500 if there's a remainder
    new_max_dimension = (quotient + 1) * 1500

    # Resize and return the image
    resized_image = cv2.resize(image, (new_max_dimension, new_max_dimension))
    return resized_image


def save_image_as_pdf(image_path, pdf_path):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Convert the image to RGB mode if it's not (necessary for some image types like PNG)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save the image as a PDF
    image.save(pdf_path, "PDF", resolution=569.0)

    print(f"Image saved as PDF at: {pdf_path}")


def convert_pdf_page_to_image(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    pix = page.get_pixmap()
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    doc.close()
    return image

def process_chunk(image1_chunk, image2_chunk):
    master_image_chunk = cv2.bitwise_and(image2_chunk, image1_chunk)

    diff_chunk = cv2.bitwise_xor(master_image_chunk, image2_chunk)
    diff_gray_chunk = cv2.cvtColor(diff_chunk, cv2.COLOR_BGR2GRAY)

    diff_chunk1 = cv2.bitwise_xor(master_image_chunk, image1_chunk)
    diff_gray_chunk1 = cv2.cvtColor(diff_chunk1, cv2.COLOR_BGR2GRAY)

    for x in range(diff_gray_chunk.shape[0]):
        for y in range(diff_gray_chunk.shape[1]):
            if diff_gray_chunk[x, y] > 225:
                master_image_chunk[x, y] = [255, 0, 0]  # Blue for differences

    for x in range(diff_gray_chunk1.shape[0]):
        for y in range(diff_gray_chunk1.shape[1]):
            if diff_gray_chunk1[x, y] > 225:
                master_image_chunk[x, y] = [0, 0, 255] #Red fir differences

    return master_image_chunk

def divide_image(image, chunk_size=(1500, 1500)):
    chunks = []
    for x in range(0, image.shape[1], chunk_size[0]):
        for y in range(0, image.shape[0], chunk_size[1]):
            chunks.append(image[y:y+chunk_size[1], x:x+chunk_size[0]])
    return chunks

def recombine_chunks(chunks, original_shape):
    # print(chunks.shape)
    new_image = np.zeros(original_shape, dtype=np.uint8)
    chunk_height, chunk_width = 1500, 1500
    for i, chunk in enumerate(chunks):
        y = (i // (original_shape[1] // chunk_width)) * chunk_height
        x = (i % (original_shape[1] // chunk_width)) * chunk_width
        new_image[x:x+chunk_width,y:y+chunk_height] = chunk
    return new_image


#comparing image using binary operations
def compare_images_with_bg_subtraction(image1, image2,output_image):
    chunks1 = divide_image(image1)
    chunks2 = divide_image(image2)
    master_image_copy = cv2.bitwise_and(image1,image2)
    num_threads = os.cpu_count()  # You might want to multiply this by a factor if tasks are I/O bound
    print(num_threads)
    processed_chunks = [None] * len(chunks1)  # Pre-allocate space for processed chunks
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_index = {executor.submit(process_chunk, chunks1[i], chunks2[i]): i for i in range(len(chunks1))}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            processed_chunk = future.result()
            processed_chunks[index] = processed_chunk

    master_image = recombine_chunks(processed_chunks, image1.shape)
    stacked_image = stack_images(master_image_copy,image1,image2,master_image)
    cv2.imwrite("output/stacked_image.jpg",stacked_image)
    cv2.imwrite(output_image, master_image)
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(master_image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    plt.show()


# Parsing command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compare two PDFs and highlight differences.")
    parser.add_argument("--pdf_path1", help="Path to the first PDF file", default='input/file_1.pdf')
    parser.add_argument("--pdf_path2", help="Path to the second PDF file", default='input/file_2.pdf')
    parser.add_argument("--output_image", help="Path for the output image", default="output/rudra.jpg")
    parser.add_argument("--output_pdf", help="Path for the output PDF", default="output/rudra.pdf")
    return parser.parse_args()


# Main execution
def main():
    args = parse_args()
    pdf_path = args.pdf_path1
    comparison_image_path = args.pdf_path2
    out_image = args.output_image
    out_pdf = args.output_pdf
    extracted_image = convert_pdf_page_to_image(pdf_path)
    comparison_image = convert_pdf_page_to_image(comparison_image_path)
    print(comparison_image.shape)
    comparison_image = resize(comparison_image)
    extracted_image = resize(extracted_image)
    print(comparison_image.shape)
    print(extracted_image.shape)
    # Compare the two images using background subtraction
    if extracted_image is not None and comparison_image is not None:
        compare_images_with_bg_subtraction(extracted_image, comparison_image,out_image)
        save_image_as_pdf(out_image, out_pdf)
    else:
        print("Error loading images.")

if __name__ == "__main__":
    main()
