import os
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import io
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def parse_pdf_multimodal(pdf_path:str, output_dir: str = "extracted_content"):
    """
    Parses a PDF document to extract text, tables, and images using Unstructured.
    Saves images to a specified output directory.
    Returns structured elements of each type.

    Args:
        pdf_path (str) :The path to the PDF file.
        output_dir (str) : Directory where extracted images and potentially table CSVs will be saved

    Returns:
        tuple: A tuple containing three lists:
            - text elements (list of dict): Each dict contains 'type', 'content', 'metadata'.
            - table elements (list of dict): Each dict contains 'type', 'content', 'metadata' (including 'html_content' if available)
            - image elements (list of dict): Each dict contains 'type', 'content' (path to image file), 'metadata'.

    """
    print(f"Parsing PDF multimodal: {pdf_path}")

    # Create o/p directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # use parition_pdf from unstructured to get various elements
    # hi_res strategy uses more advanced layout analysis for better results
    elements = partition_pdf(
        filename = pdf_path,
        strategy = "hi_res", # for better table and image extraction
        extract_images_in_pdf = True,
        infer_table_structure = True,
        output_directory = output_dir
    )

    text_elements = []
    table_elements = []
    image_elements = []

    text_file_count = 0
    table_file_count = 0

    for i, element in enumerate(elements):
        element_type = str(type(element))

        # All elements have a .text attribute, which is their plain text representation
        context_text = element.text

        # All elements have a .metadata attribute
        meta_dict = element.metadata.to_dict() if element.metadata else {}


        if 'filename' in meta_dict and 'file_directory' in meta_dict:
            #Use  'source_file' and 'source_directory' for clarity in our system
            meta_dict['source_file'] = meta_dict.pop('filename')
            meta_dict['source_directory'] = meta_dict.pop('file_directory')

        if 'page_number' in meta_dict: #making sure page_number is easily accessible
            meta_dict['page'] = meta_dict.pop('page_number')

        if 'page_name' in meta_dict:
            meta_dict['page'] = meta_dict.pop('page_name')

        # Categorize elements based on their type
        if "CompositeElement" in element_type or "NarrativeText" in element_type or "Title" in element_type or "ListItem" in element_type:
            # These are general text blocks
            text_file_count += 1
            text_filename = os.path.join(output_dir,
                                         f"text_page_{meta_dict.get('page', 'N/A')}_idx_{text_file_count}.txt")
            with open(text_filename, "w", encoding="utf-8") as f:
                f.write(context_text)
            text_elements.append({
                "type": "text",
                "content": context_text,
                "metadata": meta_dict
            })
        elif "Table" in element_type:
            table_html = getattr(element, "text_as_html", None)
            if table_html:
                meta_dict['html_content'] = table_html

            table_file_count += 1
            table_filename = os.path.join(output_dir,
                                          f"table_page_{meta_dict.get('page', 'N/A')}_idx_{table_file_count}.html")
            with open(table_filename, "w", encoding="utf-8") as f:
                f.write(table_html if table_html else context_text)

            table_elements.append({
                "type": "table",
                "content": context_text, # plain text represnetation on table data
                "metadata": meta_dict
            })
        elif "Image" in element_type:
            # For images, 'unstructured' saves them to 'output_directory' and saves the path in metadata.image_path
            image_path = meta_dict.get('image_path')
            if image_path and os.path.exists(image_path):
                image_elements.append({
                    "type":"image",
                    "content":image_path,
                    "metadata":meta_dict
                })
            else:
                print(f"Image element found but no valid image_path in metadata : {meta_dict}")

    print(f"Extracted {len(text_elements)} text elements, {len(table_elements)} table elements, {len(image_elements)} image elements.")
    return text_elements, table_elements, image_elements

if __name__=="__main__":
    pdf_path  = "/Users/truptikirve/GenAI_Projects/multimodal-rag-app-v1/scripts/raw_data/attention_is_all_you_need.pdf"
    output_dir = "./extracted_pdf_content"

    # removes any previous extractions
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"Cleaned up existing output directory: {output_dir}")

    # run the parser
    texts, tables, images = parse_pdf_multimodal(pdf_path,output_dir)

    # trying to extract sample data for inspection
    print("\n--- Sample Text Elements---")
    for i,t in enumerate(texts[:min(3, len(texts))]):
        print(f"Text {i} (Page {t['metadata'].get('page','N/A')}: {t['content'][:200]}...")

    print("\n--- Sample Table Elements---")
    for i, tbl in enumerate(tables[:min(2, len(tables))]):
        print(f"Table {i} (Page {tbl['metadata'].get('page','N/A')}): {tbl['content'][:200]}...")
        if tbl['metadata'].get('html_content'):
            print(f" HTML Content (first 200 chars): {tbl['metadata']['html_content'][:200]}....")

    print("\n-- Sample Image Elements---")
    for i, img in enumerate(images[:min(2,len(images))]):
        print(f"Image {i} (Page {img['metadata'].get('page','N/A')}): Path - {img['content']}")

    print(f"\n Extracted content saved to: {output_dir}")