import json


def recover_short_range(read, relative_path, start_line, end_line):
    raw = read(relative_path, start_line, end_line)
    error = json.loads(raw).get("error", {})
    count = error.get("total_lines")
    if error.get("code") != "line_out_of_range" or type(count) is not int or not 1 <= count <= 40:
        return raw
    result = json.loads(read(relative_path, 1, count))
    if result.get("ok") is not True:
        return raw
    data = result["data"]
    if data.get("truncated") or data.get("returned_end_line") != count:
        return raw
    data["range_recovery"] = {"requested_start_line": start_line, "requested_end_line": end_line,
                              "reason": "起始行越界，已返回短文件实际范围。"}
    return json.dumps(result, ensure_ascii=False)
