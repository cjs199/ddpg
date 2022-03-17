# 追加参数截取


def add_append_str(s, begin, end, append_s='0'):
    s += end * append_s
    return s[begin:end]

# 追加日志


def appand_log(path, log_str):
    with open(path, "a+", encoding="utf-8") as log_file:
        log_file.write(log_str)
