for i in range(19):
    file_name = "./%d.txt" % (i * 10)
    with open(file_name, "rb") as fin:
        data = fin.readlines()

    final_rst = []
    for line in data:
        line = line.decode('utf-8')
        words = line.split()
        rst = words
        for i in range(len(words)):
            if words[i] == "<EOS>":
                rst = words[:i]
                break
        final_rst.append(" ".join(rst))
    with open(file_name, "wb") as fout:
        fout.write("\n".join(final_rst).encode("utf-8"))
