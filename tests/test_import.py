def test_import():
    from myllm.core.gpt import GPTModel

    assert GPTModel is not None
