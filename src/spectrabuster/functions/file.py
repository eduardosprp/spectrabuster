import numpy as np

"""
Functions related to reading and writing text files. read_file is the
most general function, which calls any of the other "read" functions
depending on how the file is formatted.
"""

def read_file(text_file, file_type=None):
# {{{
    """
    Reads some text file. It is a wrapper for many different functions, each
    for a single file "type". It'll decide which function to call based on 1.
    the argument file_type, 2. the file extension, in this order.
    """

    # No file type specified. Will try to guess.
    if file_type is None:
        if text_file[-4:] in (".dat", ".txt"):
            return read_file(text_file, file_type="dat")
        elif text_file[-4:] == ".csv":
            return read_file(text_tile, file_type="csv")
        else:
            raise RuntimeError(
                f"No file type specified, and the file extension doesn't match any known extension."
            )

    else:
        if file_type == "dat":
            return read_dat(text_file)
        elif file_type == "csv":
            return read_csv(text_file)

        # Many more file types to be implemented...

        else:
            raise ValueError(
                f"{file_type} is not a valid file type for read_file."
            )
# }}}

def read_dat(text_file):
# {{{
    """
    Returns as many numpy arrays as there are tab-separated columns in
    the file, and a dictionary with whatever arguments (prefaced by #)
    it finds
    """

    dict_args = {}
    col1 = []
    col2 = []

    with open(text_file, "r") as arq:

        # Generator for the lines in the archive
        gen_lines = (line.split() for line in arq)

        for line_split in gen_lines:

            if (
                line_split[0] == "#"
            ):  # comment, presumably containing arguments for __init__
                dict_args[line_split[1]] = line_split[3]
            elif (
                len(line_split) > 1
            ):  # 2 or more columns. Will ignore anything after the second column
                col1.append(float(line_split[0]))
                col2.append(float(line_split[1]))
            elif len(line_split) == 1:  # 1 column
                col1.append(float(line_split[0]))

        if not dict_args and not col1 and not col2:
            # Check if they're all empty

            raise RuntimeError(
                f"No arguments, wavelengths and intensities found in {text_file}. Please check if this is a valid file."
            )

        return np.array(col1), np.array(col2), dict_args
# }}}

def read_csv(text_file):
# {{{
    raise NotImplementedError(
        "This function isn't implemented yet"
    )
# }}}
