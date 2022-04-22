from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union
import warnings

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

import re
import unicodedata

class SyllableTokenizer(AbsTokenizer):
    """
    Splits sentences by spaces and hyphens("-").
    This is written for Tailo scripts.
    """
    def __init__(
        self,
        delimiters: List,
        non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
        remove_non_linguistic_symbols: bool = False,
        atonal: bool = False,
    ):
        assert check_argument_types()
        self.delimiters = delimiters

        if not remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            warnings.warn(
                "non_linguistic_symbols is only used "
                "when remove_non_linguistic_symbols = True"
            )

        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols
        self.atonal = atonal

    def __repr__(self):
        return f'{self.__class__.__name__}(delimiters="{self.delimiters}")'

    def remove_accents(self, text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return str(text) 

    def text2tokens(self, line: str) -> List[str]:
        tokens = []
        pattern = "|".join(self.delimiters)
        for t in re.split(pattern, line):
            if self.remove_non_linguistic_symbols and t in self.non_linguistic_symbols:
                continue

            # there may be cases where there are empty strings after removing delimiters,
            # For example, in Taiwanese, "kuan--lâi."
            if t == "":
                continue

            # remove accents
            if self.atonal:
                t = self.remove_accents(t)
                
            tokens.append(t)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
        This is irreversible, sorry :(
        """
        raise NotImplementedError
