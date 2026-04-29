from pypdf import PdfReader

def extract_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    return text