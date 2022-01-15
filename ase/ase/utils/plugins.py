"""
Utilities for plugins to ase
"""

from typing import NamedTuple, Union, List, Optional


# Name is defined in the entry point
class ExternalIOFormat(NamedTuple):
    desc: str
    code: str
    module: Optional[str] = None
    glob: Optional[Union[str, List[str]]] = None
    ext: Optional[Union[str, List[str]]] = None
    magic: Optional[Union[bytes, List[bytes]]] = None
    magic_regex: Optional[bytes] = None
