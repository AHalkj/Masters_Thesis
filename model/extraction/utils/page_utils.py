from unidecode import unidecode

import fitz
import numpy as np
import cv2
import pandas as pd
import pytesseract

from svc.utils import replace_block_num
from ocr_utils import extract_bounding_boxes, LineItem


def fitz_page_transform_to_image(page):
    zoom = 8
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    image_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
    image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    return image

def lines_from_image_with_pytesseract(img_bin):
    df = pytesseract.image_to_data(img_bin, lang = "eng+ger", output_type=pytesseract.Output.DATAFRAME)

    line_text = df[df.level == 5] # .groupby(['block_num','line_num']).agg({str('text'): lambda x:' '.join(x)}).reset_index()
    line_text = line_text.assign(text = line_text.text.astype(str)).groupby(['block_num','line_num']).agg({'text': lambda x:' '.join(x)}).reset_index()
    line_pos = df[df.level == 4][['block_num', 'line_num', 'left','top', 'width','height']]
    df = pd.merge(line_pos, line_text,  how='left', on=['block_num','line_num'])
    return df

def boxAnnotation_to_dataframe_row(box, IMG_WIDTH, PAGE_WIDTH, IMG_HEIGHT, PAGE_HEIGHT):
    ymin = box.y / IMG_WIDTH * PAGE_WIDTH,
    ymax = (box.y+ box.height) / IMG_WIDTH * PAGE_WIDTH,
    xmax = (((box.x + box.width) / IMG_HEIGHT)) * PAGE_HEIGHT,
    xmin = (((box.x) / IMG_HEIGHT)) * PAGE_HEIGHT
    block_num = "t_" + str(box.row)
    row = {'xmin': xmin, 'xmax': xmax[0], 'ymin':ymin[0], 'ymax':ymax[0], 'text' : box.text, "block_num": block_num}
    return row

def span_to_dataframe_row(span, i):
    text = unidecode(span['text'])

    if text.replace(" ","") !=  "":  
        xmin, ymin, xmax, ymax = list(span['bbox'])
        font_size = round(span['size'], 2)        
        span_font = span['font']                  
        is_bold = False 
        block_num = i
        tag = 'P'
        if "bold" in span_font.lower():
            is_bold = True 
        row = (xmin, ymin, xmax, ymax, text, is_bold, span_font, font_size, block_num, tag)
        return row
    else:
        return None 

def pdf_page_classifier(page):
    res = []

    file_dict = page.get_text('dict')
    blocks = file_dict['blocks']

    text = page.get_text()
    # length of all recovered characters
    raw_text = ''.join(text.split())
    total_len=len(raw_text)
    
    alphanum_len = len([x for x in raw_text if x.isascii()])

    if len(blocks) == 1:
        if 'image' in blocks[0].keys():
            res = 'picture_scan'
    if total_len >0:
        if alphanum_len/total_len < 0.5:
            res = 'unreadable'
        else:
            res = 'normal'
    
    else: 
        res = 'unreadable'
    
        
    return res, blocks

def pdf_page_get_amount_boxes(page):
    shapes = page.get_drawings()
    shape_counter = 0
    for shape in shapes: 
        for item in shape['items']:
            if item[0] == 're':
                shape_counter = shape_counter + 1
            if item[0] == 'l':
                line = LineItem(item)
                if line.width < 1 or line.height < 1: 
                    shape_counter = shape_counter + 1
    return shape_counter

def df_from_pdf_page_ocr(page):

    image = fitz_page_transform_to_image(page)

    PAGE_WIDTH, PAGE_HEIGHT = page.rect.width, page.rect.height
    IMG_HEIGHT, IMG_WIDTH = image.shape

    
    boundingBoxes = extract_bounding_boxes(image)

    if len(boundingBoxes)>1: ## sometimes no bounding boxes are retrieved (they're not boxy enough, too small etc. then we don't create a df)
        rows = []
        for box in boundingBoxes: 

            row = boxAnnotation_to_dataframe_row(box, IMG_WIDTH, PAGE_WIDTH, IMG_HEIGHT, PAGE_HEIGHT)
            rows.append(row)

            image[box.y:box.y+box.height, box.x:box.x+box.width] = 0 
    

        df_table = pd.DataFrame(rows)

        print(page.rotation)
        if page.rotation == 90:
            df_table = df_table.assign(
                ymin = PAGE_WIDTH - df_table.xmax,
                ymax = PAGE_WIDTH - df_table.xmin ,
                xmin = df_table.ymin,
                xmax = df_table.ymax,
                tag = 't'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']] 
        if page.rotation == 180: 
            df_table = df_table.assign(
                ymin = PAGE_HEIGHT - df_table.ymax,
                ymax = PAGE_HEIGHT - df_table.ymin ,
                xmin = PAGE_WIDTH - df_table.xmax,
                xmax = PAGE_WIDTH - df_table.xmin,
                tag = 't'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']] 
        if page.rotation == 270: 
            df_table = df_table.assign(
                ymin = df_table.xmin,
                ymax = df_table.xmax ,
                xmin = PAGE_HEIGHT - df_table.ymax,
                xmax = PAGE_HEIGHT - df_table.ymin,
                tag = 't'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']] 
        else:
            df_table = df_table.assign(
                tag = 't'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']] 

    _, img_bin = cv2.threshold(image,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    
    df = lines_from_image_with_pytesseract(img_bin)

    if df is not None:

        df = df[df.text.str.strip() != '']
        if page.rotation == 90:
            df = df.assign(
            ymin = PAGE_WIDTH - ((((df.left + df.width) / IMG_HEIGHT)) * PAGE_HEIGHT),
            ymax = PAGE_WIDTH - ((((df.left) / IMG_HEIGHT)) * PAGE_HEIGHT) ,
            xmin = df.top / IMG_WIDTH * PAGE_WIDTH,
            xmax = (df.top+ df.height) / IMG_WIDTH * PAGE_WIDTH,
            tag = 'p'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']]
        if page.rotation == 270:
            df = df.assign(
            ymin = (((df.left ) / IMG_HEIGHT)) * PAGE_HEIGHT,
            ymax = (((df.left+ df.width) / IMG_HEIGHT)) * PAGE_HEIGHT ,
            xmin = PAGE_HEIGHT - (df.top+ df.height) / IMG_WIDTH * PAGE_WIDTH,
            xmax = PAGE_HEIGHT - (df.top) / IMG_WIDTH * PAGE_WIDTH,
            tag = 'p'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']]  
        if page.rotation == 180:
            df = df.assign(
            # ymin = df.top / IMG_WIDTH * PAGE_WIDTH,
            # ymax = (df.top+ df.height) / IMG_WIDTH * PAGE_WIDTH,
            # xmax = (((df.left + df.width) / IMG_HEIGHT)) * PAGE_HEIGHT,
            # xmin = (((df.left) / IMG_HEIGHT)) * PAGE_HEIGHT,
            xmin = PAGE_WIDTH - ((((df.left + df.width) / IMG_HEIGHT)) * PAGE_HEIGHT),
            xmax = PAGE_WIDTH - ((((df.left) / IMG_HEIGHT)) * PAGE_HEIGHT) ,
            ymin = PAGE_HEIGHT - ((df.top + df.height) / IMG_WIDTH * PAGE_WIDTH),
            ymax = PAGE_HEIGHT - ((df.top) / IMG_WIDTH * PAGE_WIDTH),
            tag = 'p'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']] 
        else: 
            df = df.assign(
            # ymin = df.top / IMG_WIDTH * PAGE_WIDTH,
            # ymax = (df.top+ df.height) / IMG_WIDTH * PAGE_WIDTH,
            # xmax = (((df.left + df.width) / IMG_HEIGHT)) * PAGE_HEIGHT,
            # xmin = (((df.left) / IMG_HEIGHT)) * PAGE_HEIGHT,
            ymin = (((df.left ) / IMG_HEIGHT)) * PAGE_HEIGHT,
            ymax = (((df.left + df.width) / IMG_HEIGHT)) * PAGE_HEIGHT ,
            xmin = (df.top / IMG_WIDTH * PAGE_WIDTH),
            xmax = ((df.top+ df.height) / IMG_WIDTH * PAGE_WIDTH),
            tag = 'p'
            )[['xmin','ymin','xmax','ymax','text','block_num','tag']]            
        max_block_num = len(df.block_num.value_counts())

    try: 
        df = pd.concat([df, df_table])
        df.block_num = df.block_num.apply(lambda x: replace_block_num(x, max_block_num)) #replace('table_block', max_block_num )
    except: 
        pass

    return df

def df_from_pdf_page_normal(blocks):

    rows = []

    for i, block in enumerate(blocks):
        if block['type'] == 0:
            for line in block['lines']:
                for span in line['spans']:

                    row = span_to_dataframe_row(span, i)
                    if row: 
                        rows.append(row)

                                
    page_df = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'text','is_bold','span_font', 'font_size', 'block_num', 'tag'])

    return page_df

def df_from_pdf_page(page):

    page_type, blocks = pdf_page_classifier(page)
    print(f'Type = {page_type}')

    if page_type == 'normal':
        box_counter = pdf_page_get_amount_boxes(page)

        # if more that 8 boxes are present on the page we use ocr to get the page content
        # because we assume that we have a table
        if box_counter < 8: 
            page_df = df_from_pdf_page_normal(blocks)
        else: 
            page_type = "unreadable"

    if page_type == 'unreadable' or page_type == 'picture_scan': # else
    
        page_df = df_from_pdf_page_ocr(page)

    return page_df