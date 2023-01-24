import uuid
from io import BytesIO
from pathlib import Path

import fitz
from fitz import Document, Pixmap

def convert_pixmap_to_rgb(pixmap) -> Pixmap:
    """Convert to rgb in order to write on png"""
    # check if it is already on rgb
    if pixmap.n < 4:
        return pixmap
    else:
        return fitz.Pixmap(fitz.csRGB, pixmap)

pdfs_directory_path="C:\github\dataset-cats-dogs-others"
images_directory_path="C:\github\MLOps-is-above-all-a-human-adventure\\train\extraction\images"

pdfs = [p for p in Path(pdfs_directory_path).iterdir() if p.is_file()]

for pdf_path in pdfs:
    with open(pdf_path, "rb") as pdf_stream:
        with fitz.open(stream=pdf_stream.read(), filetype="pdf") as document:
            file_images = []
            number_pages = len(document) -1
            for index in range(number_pages):
                page = document[index]
                images = document.get_page_images(index)
                number_images = len(images)
                for index_image, image in enumerate(images):
                    xref = image[0]
                    image_pix = fitz.Pixmap(document, xref)
                    image_bytes_io = BytesIO(convert_pixmap_to_rgb(image_pix).tobytes())
                    filename = pdf_path.stem + "_page" + str(index)+ "_index" + str(index_image) + ".png"
                    with open(images_directory_path + "\\" + filename, "wb") as f:
                        f.write(image_bytes_io.getbuffer())