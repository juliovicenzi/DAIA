import math

from daia.agents.models import get_embedding_model


def test_ham():
    model = get_embedding_model()
    # Somehow causes error 424 - DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")
    embeds = model.embed_documents(["Bear"])

    for emb in embeds:
        assert not any(math.isnan(x) for x in emb)
