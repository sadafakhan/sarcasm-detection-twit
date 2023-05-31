import re

def preprocess(example:str, task='adaptation')->str:
    # replaces url in example with link token to denote that it's an image file
    example = re.sub(r'http\S+', '<link>', example)

    # replaces tagged users (@ followed by twt handle) in example with empty string
    example = re.sub(r'\@\S+', '', example)

    # replaces HTTP ampersands (&amp;) with and
    example = re.sub(r'\&amp;', 'and', example)


    if task == 'adaptation':
        example = re.sub('"', '', example)

    return example