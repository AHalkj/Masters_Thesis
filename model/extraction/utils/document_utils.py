import re

import fitz
import pandas as pd

from page_utils import df_from_pdf_page

def df_from_pdf_document(path):
    doc = fitz.open(path)
    page_count = len(doc)  

    df_list = []
    for page_number, page in enumerate(doc):
        
        df = df_from_pdf_page(page).assign(page_number = page_number, page_width = page.rect.width, page_height = page.rect.height)
        df_list.append(df)
    
    df = pd.concat(df_list)

    df['is_upper'] = df.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x['text']).isupper(), axis = 1)
    try:  # these three column will be there if we have at least one OCR page in the document and occurr together
        df['span_font'] = df.span_font.fillna('N/A')
        df['is_bold'] = df.is_bold.fillna(False)
        df['font_size'] = df.font_size.fillna(df.font_size.mean())
    except: 
        df['span_font'] = 'N/A'
        df['is_bold'] = False
        df['font_size'] = 10 # random hardcoded number :D

    print(df.head(20))

    doc.close()

    return page_count, df   


if __name__ == "__main__": 
    _, df = df_from_pdf_document('rotated_table_180.pdf')
    doc = fitz.open('rotated_table_180.pdf')
    for i, row in df.iterrows():
        if 'Remarks' in row.text:
            doc[0].add_highlight_annot(fitz.Rect(row.xmin, row.ymin, row.xmax, row.ymax))
    
    doc.save('../trial.pdf')