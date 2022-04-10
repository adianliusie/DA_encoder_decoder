from .bert_embeds import BertModelExtraEmbeds, BertEmbeddingsExtra
from .longformer_embeds import LongformerModelExtraEmbeds, LongformerEmbeddingsExtra

def extra_embed(transformer, system):
    """overwrite model to have extra embeddings in inputs"""
    if system == 'bert':
        transformer.__class__ = BertModelExtraEmbeds
        transformer.embeddings.__class__ = BertEmbeddingsExtra
        transformer.embeddings.init()
        
    elif system == 'longformer':
        transformer.__class__ = LongformerModelExtraEmbeds
        transformer.embeddings.__class__ = LongformerEmbeddingsExtra
        transformer.embeddings.init()

    else:
        print('\n warning, no embeddings were added to transformer\n')
    return transformer