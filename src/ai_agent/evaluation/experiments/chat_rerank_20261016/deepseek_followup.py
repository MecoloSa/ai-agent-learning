"""同池、同时段交错比较DeepSeek与最快千问；不追加embedding或付费重排。"""
from run import run


if __name__ == "__main__":
    # 交错测试减少时段负载差异；沿用补尾规则和原始提示词，避免换模型时换评判标准。
    run(repeats=2, paid_baseline=False, limit=0, allow_partial=True,
        models=["deepseek-v4.1-flash", "qwen3.6-flash"], output_prefix="deepseek_followup_")
