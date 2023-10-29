import time
from functools import wraps
import re
from unidecode import unidecode
import os
import tempfile


import io

def bytestream_to_pdf_tempfile(bytestream):
    try:
        # Step 1: Read the bytestream into a BytesIO object
        byte_stream = io.BytesIO(bytestream)

        # Step 2: Write the contents of the BytesIO object to a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf_file:
            temp_pdf_filename = temp_pdf_file.name
            temp_pdf_file.write(byte_stream.getvalue())

        # Step 4: Close and remove the temporary file (this will delete it)
        temp_pdf_file.close()

        return temp_pdf_filename, temp_pdf_file

    except Exception as e:
        print("Error:", e)
        return None


if os.getenv('DJANGO_ALLOWED_HOSTS') == 'localhost':
    from pywintypes import com_error
    import pythoncom
    import win32com.client

signal_words = ["best regards", "signature","page", "managment fee", "dear", "kind regards", "personal data", "vote", "minimum commitment", "onboarding fee", "prospective investor"]
counter_signal_words = ["shall not invest", "will not invest", "is authorised to invest", "the fund invests at least", "does not invest more", "cannot invest more"]

 # ....... Helper Functions ........


def replace_block_num(block_num, max_len):
    if isinstance(block_num, int):
        b = block_num
    else: 
        b = int(block_num.replace('t_', '')) + max_len + 1
    return b


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

