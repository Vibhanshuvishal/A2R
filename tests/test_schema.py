import pytest
from pydantic import ValidationError

from a2r.serving.schema import QueryRequest


def test_blank_and_oversize_queries_are_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")
    with pytest.raises(ValidationError):
        QueryRequest(query="x" * 2001)
