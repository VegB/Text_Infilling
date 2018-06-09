def print_and_write_to_file(rst, fout, print_out=True):
    if print_out:
        print(rst)
    fout.write(rst)
    fout.flush()