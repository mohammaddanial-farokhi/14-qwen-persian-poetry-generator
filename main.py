import hazm
import re

# def num_normalizer(text):
#     """In Persian poetry, numbers are insignificant and should be removed."""

#     _normalizer = hazm.Normalizer(persian_numbers=False)

#     normall_text = _normalizer.normalize(text)
#     return normall_text


def num_normalizer(text: str) -> str:
    """Remove numbers as they are not important, and normalize the text."""

    _normalizer = hazm.Normalizer(persian_numbers=False)
    normalized_text = _normalizer.normalize(text)
    no_digits_text = re.sub(
        r"[0-9۰-۹٠-٩]+", "", normalized_text
    )  # latin, persian, arabic numbers
    return no_digits_text.strip()


if __name__ == "__main__":
    print()
    print(num_normalizer("تست ي ي میخواهم می   خاهم می خواهم دو سه 2 3  "))
