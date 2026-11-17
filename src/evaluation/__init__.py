"""Đánh giá hiệu năng: Coverage, Precision, Factual Error, Human Rating, Agreement, Wilcoxon và Report."""


def __getattr__(name: str):
    if name == "generate_report":
        from .report import generate_report

        return generate_report
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


