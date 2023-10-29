import argparse
from svc.utils import timeit, pdf_extract, convert_xlsx, convert_docx
import os
import time
import re
from svc.utils import countries

DOC_TYPES = ['xlsx', 'pdf', 'docx']

class DocumentClass():

    def __init__(self, path):
        self.path = path
        ending = path.split('.')[-1] 
        if ending in DOC_TYPES:
            self.type = ending

    @timeit
    def extract(self):
        if self.type == 'xlsx':

            print('Extracting Excel Document')

            # self.page_count = xlsx_extract(self.path)
            path = self.path.replace('.xlsx', '.pdf')
            convert_xlsx(os.path.abspath(self.path))
            self.path = path
            self.type = 'pdf'
            time.sleep(2)
            return (self.extract())

        elif self.type == 'docx': 

            print('Extracting Word Document')

            path = self.path.replace('.docx', '.pdf')
            convert_docx(os.path.abspath(self.path))
            self.path = path
            self.type = 'pdf'

            return (self.extract())

        elif self.type == 'pdf':

            print(f'Extracting PDF Document from {self.path}')

            self.page_count, df = pdf_extract(self.path)
            return df

    def __len__(self):
        return self.page_count or None
    
    def pre_label(self):
        ## Option 1: Extract EMail Adresses: 
        ## Option 2: Extract IBANs: 
        ## Option 3: Extract Page Numbers
        ## Option 4: Extract Larger than usual
        ## Option 5: Extract Smaller than usual
        return self.df
    
    def anonymize_text(self, company_name : str = None):
        #to replace: [COMPANY], [IBAN], [COUNTRY], [PHONENR], [DATE]
        country_regex = r"\b(?:{})\b".format('|'.join(map(re.escape, countries)))
        df['anonymized_text'] = (df.text.str
                                 .replace(company_name, '[COMPANY]')
                                 .replace(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', '[EMAIL]')
                                 .replace(r'^(?=[A-Za-z]{2})(?:[A-Z0-9]{2})(?:(?:(?:[0-9A-Za-z]{4}(?!$)){1,4}(?:[0-9A-Za-z]{1,4}))|(?:[0-9A-Za-z]{1,4}))$', '[IBAN]')
                                 .replace(r'^(?:\+\d{1,3}\s?)?(?:\(\d{1,3}\)|\d{1,3})[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}$', '[PHONENR]')
                                 )
        df['anonymized_text'] = df.anonymized_text.apply(lambda x: re.sub(country_regex, '[COUNTRY]', str(x)))
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str)
    args = parser.parse_args()

    path = args.path
    
    doc = DocumentClass(path)
    df = doc.extract()
    df = doc.anonymize_text('T. Rowe Price')
    print(df[df.page_num == 24].anonymized_text.to_csv('bla.csv'))